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

void decode_space(memory_space_t &space, ptx_thread_info *thread,
                  const operand_info &op, memory_space *&mem, addr_t &addr);

void my_cuda_printf(const char *fmtstr, const char *arg_list) {
  FILE *fp = stdout;
  unsigned i = 0, j = 0;
  unsigned arg_offset = 0;
  char buf[64];
  bool in_fmt = false;
  while (fmtstr[i]) {
    char c = fmtstr[i++];
    if (!in_fmt) {
      if (c != '%') {
        fprintf(fp, "%c", c);
      } else {
        in_fmt = true;
        buf[0] = c;
        j = 1;
      }
    } else {
      if (!(c == 'u' || c == 'f' || c == 'd')) {
        printf(
            "GPGPU-Sim PTX: ERROR ** printf parsing support is limited to %%u, "
            "%%f, %%d at present");
        abort();
      }
      buf[j] = c;
      buf[j + 1] = 0;
      void *ptr = (void *)&arg_list[arg_offset];
      // unsigned long long value = ((unsigned long long*)arg_list)[arg_offset];
      if (c == 'u' || c == 'd') {
        fprintf(fp, buf, *((unsigned long long *)ptr));
      } else if (c == 'f') {
        double tmp = *((double *)ptr);
        fprintf(fp, buf, tmp);
      }
      arg_offset++;
      in_fmt = false;
    }
  }
}

void gpgpusim_cuda_vprintf(const ptx_instruction *pI, ptx_thread_info *thread,
                           const function_info *target_func) {
  char *fmtstr = NULL;
  char *arg_list = NULL;
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  assert(n_args == 2);
  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);
    const symbol *formal_param = target_func->get_arg(arg);
    unsigned size = formal_param->get_size_in_bytes();
    assert(formal_param->is_param_local());
    assert(actual_param_op.is_param_local());
    addr_t from_addr = actual_param_op.get_symbol()->get_address();
    unsigned long long buffer[1024];
    assert(size < 1024 * sizeof(unsigned long long));
    thread->m_local_mem->read(from_addr, size, buffer);
    addr_t addr =
        (addr_t)buffer[0];  // should be pointer to generic memory location
    memory_space *mem = NULL;
    memory_space_t space = generic_space;
    decode_space(space, thread, actual_param_op, mem,
                 addr);  // figure out which space
    if (arg == 0) {
      unsigned len = 0;
      char b = 0;
      do {  // figure out length
        mem->read(addr + len, 1, &b);
        len++;
      } while (b);
      fmtstr = (char *)malloc(len + 64);
      for (int i = 0; i < len; i++) mem->read(addr + i, 1, fmtstr + i);
      // mem->read(addr,len,fmtstr);
    } else {
      unsigned len = thread->get_finfo()->local_mem_framesize();
      arg_list = (char *)malloc(len + 64);
      for (int i = 0; i < len; i++) mem->read(addr + i, 1, arg_list + i);
      // mem->read(addr,len,arg_list);
    }
  }
  my_cuda_printf(fmtstr, arg_list);
  free(fmtstr);
  free(arg_list);
}


// yasai: reads the value of parameter arg of a call (as gpgpusim_cuda_vprintf does).
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
