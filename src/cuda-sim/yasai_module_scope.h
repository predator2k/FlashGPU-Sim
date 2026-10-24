// yasai: per-module scope for PTX symbols without linkage.
//
// The simulator keeps one symbol table for all PTX modules of a process, so two modules that
// define the same module-local name (nvcc's "$str", "$str$1", ... for printf/assert strings,
// static __device__ / __constant__ / __shared__ variables, internal .func helpers) shared one
// symbol: the second module's declaration was skipped and its kernels read the first module's
// data (UCCL-EP's device assert printed a libtorch_cuda string). A module-local symbol is a
// file-scope .global, .const or .shared variable, or a .func, declared without .visible,
// .extern, .weak or .common. scope_file() gives every such name of one PTX file a suffix that
// is unique to the file ("__ym" and a hash of the file's base name, the same in every process
// and run), in a copy of the file; .visible, .extern and .weak names keep resolving across
// modules by name. local_name() maps a name that the host registers for a module
// (__cudaRegisterVar of a static device variable) to the module's renamed symbol.
#ifndef YASAI_MODULE_SCOPE_H_INCLUDED
#define YASAI_MODULE_SCOPE_H_INCLUDED

#include <string>

namespace yasai_scope {

/// Path of the PTX file to parse in place of path: path itself when the file declares no
/// module-local symbol (or YASAI_MODULE_SCOPE=0), otherwise a copy in the directory
/// yasai_module_scope/ of the working directory with every module-local name renamed. The text
/// keeps its lines, so line-keyed information (SASS guides, debug output) still applies. A path
/// is scanned once per process; later calls return the same copy.
/// @param path PTX file.
/// @return The file to parse.
std::string scope_file(const std::string &path);

/// The renamed symbol for a module-local name registered by the host, or name itself when no
/// module renamed it. The candidate whose file has the library key of library (the fat
/// binary's file; "libfoo" for "libfoo.so", "libfoo.so.1.sm_120.ptx" and "libfoo.1.sm_120a.ptx")
/// is chosen. Without a match, a single candidate is chosen unless some module defines the name
/// with linkage; otherwise the name is returned unchanged.
/// @param name Name as the host registers it.
/// @param library File name of the fat binary that registers it ("" when unknown).
/// @return The name to look up.
std::string local_name(const std::string &name, const std::string &library);

}  // namespace yasai_scope

#endif
