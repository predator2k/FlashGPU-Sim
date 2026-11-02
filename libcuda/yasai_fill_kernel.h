// yasai: host-executed fill kernels (YASAI_FILL_KERNEL_HOST=1).
//
// PyTorch fills a contiguous tensor (fill_, zero_, full, ones, the L2 flush of Inductor's
// benchmark_gpu) with at::native::vectorized_elementwise_kernel<vec, FillFunctor<T>, array<char*,1>>
// or, when the destination is not aligned for vector stores, unrolled_elementwise_kernel<
// FillFunctor<T>, array<char*,1>, n, TrivialOffsetCalculator<0>, TrivialOffsetCalculator<1>,
// LoadWithoutCast, StoreWithoutCast>. Both store FillFunctor's value unchanged into elements
// [0, N) of the destination. In the functional simulator such a kernel can take minutes (65536
// CTAs for an L2-sized buffer), so with the option on it is executed as a host-side write of N
// copies of the value's bytes at the destination, at the kernel's place in its stream.
//
// Recognition uses only the mangled kernel name (kernel family, FillFunctor's type, and for the
// unrolled kernel the offset calculators and the no-cast loader and storer) and the launch's
// parameters: the PTX entry and the launch must both have the family's parameter count and
// sizes (N: 4 bytes; the functor: sizeof(T); the pointer array: 8 bytes; the unrolled kernel's
// four empty helper objects: 1 byte each). Grid and size never select the path. A name of the
// two families with FillFunctor whose layout does not match runs in full, with one log line.
#ifndef YASAI_FILL_KERNEL_H_INCLUDED
#define YASAI_FILL_KERNEL_H_INCLUDED

#include <cstdint>
#include <string>

#include "../src/abstract_hardware_model.h"

class function_info;

namespace yasai_fill {

/// Outcome of recognize for one launch.
struct Plan {
  bool candidate = false;  ///< the name is a vectorized or unrolled FillFunctor kernel
  bool host = false;       ///< the launch is executed as a host write (layout matched)
  uint64_t dst = 0;        ///< device address of element 0 (the pointer array's entry)
  uint64_t count = 0;      ///< element count N (the kernel's first parameter)
  std::string pattern;     ///< bytes of one element (FillFunctor's value, sizeof(T) bytes)
  std::string type;        ///< FillFunctor's type as mangled in the name (for example "i")
  std::string reason;      ///< why a candidate runs in full (empty when host is true)
};

/// True when YASAI_FILL_KERNEL_HOST is set to a nonzero value (read once).
bool enabled();

/// Classifies a launch.
/// @param name Mangled kernel name (the PTX entry name, before any module-scope renaming).
/// @param entry The kernel's function_info (its PTX parameter list), or NULL.
/// @param args The launch's arguments as kernel_config keeps them (last parameter first).
/// @return The plan; host is true only when every check passed.
Plan recognize(const std::string &name, function_info *entry, const gpgpu_ptx_sim_arg_list_t &args);

}  // namespace yasai_fill

#endif
