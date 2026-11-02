// yasai_fill_kernel.cc: see yasai_fill_kernel.h.
#include "yasai_fill_kernel.h"

#include <cstdlib>
#include <cstring>
#include <regex>
#include <vector>

#include "../src/cuda-sim/ptx_ir.h"
#include "../src/cuda-sim/yasai_module_scope.h"

namespace yasai_fill {

namespace {

/// Size in bytes and PyTorch dtype name of FillFunctor's type T, from its Itanium mangling;
/// size 0 for a type not listed (such a kernel runs in full).
/// @param t The mangled type (for example "i", "N3c104HalfE", "N3c107complexIfEE").
/// @param dtype Set to the dtype name (for example "int32", "float16", "float8_e4m3fn").
size_t type_size(const std::string &t, std::string *dtype) {
  if (t.size() == 1) {
    switch (t[0]) {
      case 'a': *dtype = "int8"; return 1;
      case 'b': *dtype = "bool"; return 1;
      case 'c': *dtype = "char"; return 1;
      case 'h': *dtype = "uint8"; return 1;
      case 's': *dtype = "int16"; return 2;
      case 't': *dtype = "uint16"; return 2;
      case 'i': *dtype = "int32"; return 4;
      case 'j': *dtype = "uint32"; return 4;
      case 'f': *dtype = "float32"; return 4;
      case 'l': case 'x': *dtype = "int64"; return 8;
      case 'm': case 'y': *dtype = "uint64"; return 8;
      case 'd': *dtype = "float64"; return 8;
      default: return 0;
    }
  }
  static const std::regex float8("N3c10[0-9]+Float8_([a-z0-9]+)E");
  static const std::regex complex16("N3c107complexINS[0-9A-Z]*_(4Half|8BFloat16)EEE");
  std::smatch m;
  if (t == "N3c104HalfE") { *dtype = "float16"; return 2; }
  if (t == "N3c108BFloat16E") { *dtype = "bfloat16"; return 2; }
  if (std::regex_match(t, m, float8)) { *dtype = "float8_" + m[1].str(); return 1; }
  if (t == "N3c107complexIfEE") { *dtype = "complex64"; return 8; }
  if (t == "N3c107complexIdEE") { *dtype = "complex128"; return 16; }
  if (std::regex_match(t, m, complex16)) {
    *dtype = m[1].str() == "4Half" ? "complex32" : "complex_bfloat16";
    return 4;
  }
  return 0;
}

/// The launch's arguments in parameter order (kernel_config keeps the last parameter first).
std::vector<const gpgpu_ptx_sim_arg *> in_order(const gpgpu_ptx_sim_arg_list_t &args) {
  std::vector<const gpgpu_ptx_sim_arg *> v;
  for (const gpgpu_ptx_sim_arg &a : args) v.insert(v.begin(), &a);
  return v;
}

}  // namespace

bool enabled() {
  static const bool on = [] {
    const char *e = getenv("YASAI_FILL_KERNEL_HOST");
    return e != NULL && *e != '\0' && atoi(e) != 0;
  }();
  return on;
}

Plan recognize(const std::string &mangled, function_info *entry, const gpgpu_ptx_sim_arg_list_t &args) {
  Plan p;
  const std::string name = yasai_scope::source_name(mangled);
  // Kernel family and FillFunctor's type T; expected parameter sizes (0 stands for sizeof(T)).
  static const std::regex vectorized(
      "_ZN2at6native29vectorized_elementwise_kernelILi[0-9]+ENS0_11FillFunctorI(.+)EESt5array"
      "IPcLm1EE(Lb[01]E)?EEviT0_T1_");
  static const std::regex unrolled(
      "_ZN2at6native27unrolled_elementwise_kernelINS0_11FillFunctorI(.+)EESt5arrayIPcLm1EELi"
      "[0-9]+E23TrivialOffsetCalculatorILi0EjES[0-9A-Z]*_ILi1EjENS0_6memory15LoadWithoutCast"
      "ENS[0-9A-Z]*_16StoreWithoutCastEEEviT_T0_T2_T3_T4_T5_");
  std::smatch m;
  std::vector<size_t> want;
  if (std::regex_match(name, m, vectorized)) {
    want = {4, 0, 8};
  } else if (std::regex_match(name, m, unrolled)) {
    want = {4, 0, 8, 1, 1, 1, 1};
  } else {
    return p;  // not a fill kernel of the two families
  }
  p.candidate = true;
  p.type = m[1].str();
  const size_t tsize = type_size(p.type, &p.dtype);
  p.elem_bytes = tsize;
  if (tsize == 0) {
    p.reason = "FillFunctor type " + p.type + " not in the size table";
    return p;
  }
  want[1] = tsize;
  if (entry == NULL) {
    p.reason = "no PTX entry";
    return p;
  }
  // The PTX entry's declared parameters.
  if (entry->yasai_param_count() != want.size()) {
    p.reason = "PTX entry has " + std::to_string(entry->yasai_param_count()) + " parameters, expected " +
               std::to_string(want.size());
    return p;
  }
  for (unsigned i = 0; i < want.size(); ++i)
    if (entry->yasai_param_size(i) != want[i]) {
      p.reason = "PTX parameter " + std::to_string(i) + " has " +
                 std::to_string(entry->yasai_param_size(i)) + " bytes, expected " +
                 std::to_string(want[i]);
      return p;
    }
  // The launch's arguments.
  const std::vector<const gpgpu_ptx_sim_arg *> a = in_order(args);
  if (a.size() != want.size()) {
    p.reason = "launch has " + std::to_string(a.size()) + " arguments, expected " +
               std::to_string(want.size());
    return p;
  }
  for (unsigned i = 0; i < want.size(); ++i)
    if (a[i]->m_nbytes != want[i] || a[i]->m_start == NULL) {
      p.reason = "launch argument " + std::to_string(i) + " has " + std::to_string(a[i]->m_nbytes) +
                 " bytes, expected " + std::to_string(want[i]);
      return p;
    }
  int32_t n = 0;
  memcpy(&n, a[0]->m_start, sizeof(n));
  if (n < 0) {
    p.reason = "negative element count " + std::to_string(n);
    return p;
  }
  uint64_t dst = 0;
  memcpy(&dst, a[2]->m_start, sizeof(dst));
  if (dst == 0 && n > 0) {
    p.reason = "null destination";
    return p;
  }
  p.count = uint64_t(n);
  p.dst = dst;
  p.pattern.assign(static_cast<const char *>(a[1]->m_start), tsize);
  p.host = true;
  return p;
}

}  // namespace yasai_fill
