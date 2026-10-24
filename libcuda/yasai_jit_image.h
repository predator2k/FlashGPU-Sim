// yasai: PTX for modules that a process hands to the driver at run time.
//
// The simulator executes PTX. Libraries loaded with dlopen after start-up (FlashInfer and
// sglang JIT modules, framework extensions not named in PYTORCH_BIN) register their fat
// binaries with __cudaRegisterFatBinary, and compilers that run inside the process (Triton,
// DeepGEMM, CuTe DSL) pass a module image to cuModuleLoadData. This module finds the PTX of both:
// * a library file: the PTX files that `cuobjdump -xptx all` extracts from it;
// * a module image: a PTX image is used as is; a fat binary is resolved through its embedded PTX;
//   an ELF cubin (no PTX inside) is resolved through the JIT caches, first by finding a cached
//   .cubin with the same bytes and taking the .ptx next to it (Triton keeps <name>.cubin and
//   <name>.ptx in one cache directory), then by finding a cached .ptx file that defines every
//   kernel of the cubin.
// Environment:
//   YASAI_JIT_PTX_DIRS  colon-separated cache directories searched for cubin images (default
//                       $TRITON_CACHE_DIR or $HOME/.triton/cache, plus $HOME/.cache/flashinfer and
//                       $HOME/.deep_gemm); every directory is searched recursively.
//   YASAI_JIT_WORK      directory for extracted and written PTX files (default
//                       $TMPDIR/yasai-jit-<pid>).
//   YASAI_JIT_DEBUG     nonzero: print how each image was resolved.
#ifndef YASAI_JIT_IMAGE_H
#define YASAI_JIT_IMAGE_H

#include <string>
#include <vector>

namespace yasai_jit {

/// PTX file for a module image given to cuModuleLoadData, or "" when none is found.
///   image           the image pointer (PTX text, fat binary, fat binary wrapper or ELF cubin)
///   max_capability  highest PTX architecture to select (0: the highest available)
///   how             set to a one-line description of the resolution (for the log)
std::string ptx_for_image(const void *image, unsigned max_capability, std::string *how);

/// PTX files of a library or module file (one selected architecture per file), or empty.
///   path            the .so, .cubin or .fatbin file
///   max_capability  highest PTX architecture to select (0: the highest available)
std::vector<std::string> ptx_files_of_library(const std::string &path, unsigned max_capability);

/// Architecture ("sm_XX", with suffix letters) named in an extracted PTX file name, or "".
///   ptx_file        a file name of the form <name>.<n>.sm_<arch>.ptx
std::string arch_of_ptx_file(const std::string &ptx_file);

}  // namespace yasai_jit

#endif  // YASAI_JIT_IMAGE_H
