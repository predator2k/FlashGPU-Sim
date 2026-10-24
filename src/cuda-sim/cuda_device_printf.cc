// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "cuda_device_printf.h"
#include "ptx_ir.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

void decode_space(memory_space_t &space, ptx_thread_info *thread,
                  const operand_info &op, memory_space *&mem, addr_t &addr);

// yasai: reads the value of parameter arg of a call (the parameter buffer of the call).
//   pI: the call; thread: calling thread; target_func: callee; arg: parameter index
static unsigned long long yasai_call_param(const ptx_instruction *pI, ptx_thread_info *thread,
                                           const function_info *target_func, unsigned arg) {
  const operand_info &actual_param_op = pI->operand_lookup(target_func->has_return() + 1 + arg);
  const symbol *formal_param = target_func->get_arg(arg);
  unsigned long long buffer[8] = {0};
  unsigned size = formal_param->get_size_in_bytes();
  if (size > sizeof(buffer)) size = sizeof(buffer);
  if (actual_param_op.is_param_local())
    thread->m_local_mem->read(actual_param_op.get_symbol()->get_address(), size, buffer);
  return buffer[0];
}

// yasai: a NUL-terminated string at a generic address of the calling thread (at most n bytes).
//   pI: the call; thread: calling thread; target_func: callee; arg: parameter index; n: limit
static std::string yasai_call_string(const ptx_instruction *pI, ptx_thread_info *thread,
                                     const function_info *target_func, unsigned arg, unsigned n) {
  addr_t addr = (addr_t)yasai_call_param(pI, thread, target_func, arg);
  if (addr == 0) return "(null)";
  const operand_info &actual_param_op = pI->operand_lookup(target_func->has_return() + 1 + arg);
  memory_space *mem = NULL;
  memory_space_t space = generic_space;
  decode_space(space, thread, actual_param_op, mem, addr);
  std::string out;
  for (unsigned i = 0; i < n; i++) {
    char b = 0;
    mem->read(addr + i, 1, &b);
    if (b == 0) break;
    out.push_back(b);
  }
  return out;
}

// yasai: __assertfail(message, file, line, function, charSize): prints the device assertion of
// the calling thread as the CUDA driver does (the first 16 per process).
//   pI: the call; thread: calling thread; target_func: the declared __assertfail
void yasai_cuda_assertfail(const ptx_instruction *pI, ptx_thread_info *thread,
                           const function_info *target_func) {
  static unsigned printed = 0;
  if (printed >= 16 || target_func->num_args() < 4) return;
  printed++;
  const std::string msg = yasai_call_string(pI, thread, target_func, 0, 512);
  const std::string file = yasai_call_string(pI, thread, target_func, 1, 512);
  const unsigned line = (unsigned)yasai_call_param(pI, thread, target_func, 2);
  const std::string func = yasai_call_string(pI, thread, target_func, 3, 512);
  const dim3 cta = thread->get_ctaid();
  const dim3 tid = thread->get_tid();
  printf("GPGPU-Sim PTX: device assert: %s:%u: %s: block: [%u,%u,%u], thread: [%u,%u,%u] "
         "Assertion `%s` failed.\n",
         file.c_str(), line, func.c_str(), cta.x, cta.y, cta.z, tid.x, tid.y, tid.z, msg.c_str());
  fflush(stdout);
}

// yasai: device printf. vprintf(format, args) receives a generic pointer to the format string and
// a generic pointer to the argument buffer (0 when there are no arguments). The buffer follows the
// CUDA ABI: every argument is aligned to its own size; char and short are promoted to 4-byte int,
// float to 8-byte double, and long, long long, size_t and pointers take 8 bytes. %s reads a
// NUL-terminated string at a generic address (global, shared or local). The full C conversion set
// is formatted with the host's snprintf: flags, width and precision (also '*', taken from the
// buffer as int), the length modifiers hh, h, l, ll, j, z, t and L, and the conversions d i u o x X
// c s p f F e E g G a A and %%. %n writes nothing.

// yasai: copies n bytes at a generic address of the calling thread into out; false for address 0.
//   thread: calling thread; op: an operand of the call (for decode_space); addr: generic address;
//   out: destination; n: bytes
static bool yasai_read_generic(ptx_thread_info *thread, const operand_info &op, addr_t addr,
                               void *out, unsigned n) {
  if (addr == 0) return false;
  memory_space *mem = NULL;
  memory_space_t space = generic_space;
  decode_space(space, thread, op, mem, addr);
  mem->read(addr, n, out);
  return true;
}

// yasai: the NUL-terminated string at a generic address (at most limit bytes), "(null)" for 0.
//   thread: calling thread; op: an operand of the call; addr: generic address; limit: maximum
static std::string yasai_read_generic_string(ptx_thread_info *thread, const operand_info &op,
                                             addr_t addr, unsigned limit) {
  if (addr == 0) return "(null)";
  memory_space *mem = NULL;
  memory_space_t space = generic_space;
  decode_space(space, thread, op, mem, addr);
  std::string s;
  for (unsigned i = 0; i < limit; i++) {
    char b = 0;
    mem->read(addr + i, 1, &b);
    if (b == 0) break;
    s.push_back(b);
  }
  return s;
}

// yasai: formats a device printf call (see above) and returns the text.
//   fmt: the format string; thread: calling thread; op: an operand of the call; args: generic
//   address of the argument buffer (0: none)
static std::string yasai_format_device_printf(const std::string &fmt, ptx_thread_info *thread,
                                              const operand_info &op, addr_t args) {
  std::string out;
  size_t offset = 0;  // next byte of the argument buffer
  // next argument of size bytes (4 or 8), aligned to its size
  auto next_arg = [&](unsigned size) -> unsigned long long {
    offset = (offset + size - 1) / size * size;
    unsigned long long v = 0;
    if (args != 0) yasai_read_generic(thread, op, args + offset, &v, size);
    offset += size;
    return v;
  };
  size_t i = 0;
  while (i < fmt.size()) {
    const char c = fmt[i];
    if (c != '%') {
      out.push_back(c);
      ++i;
      continue;
    }
    if (i + 1 < fmt.size() && fmt[i + 1] == '%') {
      out.push_back('%');
      i += 2;
      continue;
    }
    // %[flags][width][.precision][length]conversion
    std::string spec = "%";
    size_t j = i + 1;
    while (j < fmt.size() && strchr("-+ #0'", fmt[j])) spec.push_back(fmt[j++]);
    if (j < fmt.size() && fmt[j] == '*') {
      spec += std::to_string(int(next_arg(4)));
      ++j;
    } else {
      while (j < fmt.size() && isdigit((unsigned char)fmt[j])) spec.push_back(fmt[j++]);
    }
    if (j < fmt.size() && fmt[j] == '.') {
      spec.push_back(fmt[j++]);
      if (j < fmt.size() && fmt[j] == '*') {
        spec += std::to_string(int(next_arg(4)));
        ++j;
      } else {
        while (j < fmt.size() && isdigit((unsigned char)fmt[j])) spec.push_back(fmt[j++]);
      }
    }
    std::string len;
    while (j < fmt.size() && strchr("hljztL", fmt[j])) len.push_back(fmt[j++]);
    if (j >= fmt.size()) {  // incomplete specification: printed as it is
      out.append(fmt, i, std::string::npos);
      break;
    }
    const char conv = fmt[j++];
    // 8-byte integers for l, ll, j, z and t; 4-byte int (promoted char/short) otherwise
    const bool wide = len == "l" || len == "ll" || len == "j" || len == "z" || len == "t";
    char buf[512];
    int n = 0;
    switch (conv) {
      case 'd':
      case 'i': {
        long long v = wide ? (long long)next_arg(8) : (long long)(int)next_arg(4);
        if (len == "hh") v = (signed char)v;
        if (len == "h") v = (short)v;
        n = snprintf(buf, sizeof buf, (spec + "ll" + conv).c_str(), v);
        break;
      }
      case 'u':
      case 'o':
      case 'x':
      case 'X': {
        unsigned long long v = wide ? next_arg(8) : (unsigned long long)(unsigned)next_arg(4);
        if (len == "hh") v &= 0xffull;
        if (len == "h") v &= 0xffffull;
        n = snprintf(buf, sizeof buf, (spec + "ll" + conv).c_str(), v);
        break;
      }
      case 'c':
        n = snprintf(buf, sizeof buf, (spec + conv).c_str(), (int)next_arg(4));
        break;
      case 's': {
        const std::string s = yasai_read_generic_string(thread, op, (addr_t)next_arg(8), 65536);
        out += [&] {  // a long string is formatted into a buffer of its own size
          std::vector<char> big(s.size() + 512);
          snprintf(big.data(), big.size(), (spec + "s").c_str(), s.c_str());
          return std::string(big.data());
        }();
        i = j;
        continue;
      }
      case 'p':
        n = snprintf(buf, sizeof buf, (spec + "p").c_str(), (void *)(uintptr_t)next_arg(8));
        break;
      case 'f':
      case 'F':
      case 'e':
      case 'E':
      case 'g':
      case 'G':
      case 'a':
      case 'A': {
        const unsigned long long bits = next_arg(8);
        double d;
        memcpy(&d, &bits, sizeof d);
        n = snprintf(buf, sizeof buf, (spec + conv).c_str(), d);
        break;
      }
      case 'n':
        next_arg(8);
        break;
      default:  // unknown conversion: printed as it is
        out.append(fmt, i, j - i);
        i = j;
        continue;
    }
    if (n > 0) out.append(buf, std::min<size_t>(size_t(n), sizeof buf - 1));
    i = j;
  }
  return out;
}

void gpgpusim_cuda_vprintf(const ptx_instruction *pI, ptx_thread_info *thread,
                           const function_info *target_func) {
  assert(target_func->num_args() == 2);
  const operand_info &fmt_op = pI->operand_lookup(target_func->has_return() + 1);
  const addr_t fmt_addr = (addr_t)yasai_call_param(pI, thread, target_func, 0);
  const addr_t args = (addr_t)yasai_call_param(pI, thread, target_func, 1);
  const std::string fmt = yasai_read_generic_string(thread, fmt_op, fmt_addr, 65536);
  const std::string text = yasai_format_device_printf(fmt, thread, fmt_op, args);
  fwrite(text.data(), 1, text.size(), stdout);
  fflush(stdout);
  // The int that vprintf returns (the number of arguments) is not written; callers of printf in
  // device code do not use it.
}
