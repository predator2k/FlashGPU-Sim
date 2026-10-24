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
//
// Kernels (.entry) have linkage, but separate modules may still define the same kernel name
// with different bodies (two Inductor graphs both compile a "triton_poi_fused_..._0"; two
// libraries instantiate the same template kernel). The first unit that defines a kernel name
// keeps it; a file of another unit gets the kernel renamed with its module suffix. A unit is a
// library (all PTX files with one library key: the architecture variants of one library share
// their kernels, the first definition winning as before) or one module loaded by the driver API
// (its PTX path), or one translation unit of a library or program when the caller knows its fat
// binary (yasai_fatbin_units.h): two translation units of one library may define the same
// template kernel with different bodies. module_entry_name() maps a kernel name for a driver-API module
// (cuModuleGetFunction), library_entry_name() for a fat binary's library (__cudaRegisterFunction),
// and source_name() maps a renamed kernel back to its source name (records, cuKernelGetName).
#ifndef YASAI_MODULE_SCOPE_H_INCLUDED
#define YASAI_MODULE_SCOPE_H_INCLUDED

#include <string>

namespace yasai_scope {

/// Path of the PTX file to parse in place of path: path itself when the file declares no
/// module-local symbol (or YASAI_MODULE_SCOPE=0), otherwise a copy with every module-local name
/// renamed, named <hash of the source>-<base name>, in YASAI_MODULE_SCOPE_DIR (a cache shared by
/// runs) or in yasai_module_scope/ of the working directory. The text
/// keeps its lines, so line-keyed information (SASS guides, debug output) still applies. A path
/// is scanned once per process; later calls return the same copy.
/// @param path PTX file.
/// @param unit Unit that owns the file's kernels: "" for a file of a library or program (the
///        file name without its index and architecture parts), library_unit() of the library
///        when the caller knows it, the PTX path for a module loaded through the driver API.
/// @return The file to parse.
std::string scope_file(const std::string &path, const std::string &unit = "");

/// The symbol of kernel name in the module loaded from path (scope_file's path): the renamed
/// symbol when the module's kernel was renamed, otherwise name.
/// @param path PTX file of the module, as passed to scope_file.
/// @param name Kernel name as the program asks for it.
/// @return The name to look up.
std::string module_entry_name(const std::string &path, const std::string &name);

/// The symbol of kernel name for a fat binary of library: the renamed symbol when that
/// library's file renamed the kernel, otherwise name.
/// @param name Kernel name as the host registers it.
/// @param library File name of the fat binary that registers it ("" when unknown).
/// @return The name to look up.
std::string library_entry_name(const std::string &name, const std::string &library);

/// The symbol of kernel name for the unit that registers it (a translation unit "<library
/// key>#<fat binary number>" from yasai_fatbin_units.h, as passed to scope_file): the renamed
/// symbol when that unit's file renamed the kernel, otherwise name (the unit that defined the
/// name first keeps it).
/// @param name Kernel name as the host registers it.
/// @param unit Unit of the registering fat binary ("" when unknown).
/// @return The name to look up.
std::string unit_entry_name(const std::string &name, const std::string &unit);

/// Unit of a library's PTX files for scope_file: the library key of the library's file name
/// ("libfoo" for "/path/libfoo.so.1"). Files extracted from a library may be named after its
/// source files, so a caller that knows the library passes this unit.
/// @param library Path or file name of the library.
/// @return The unit.
std::string library_unit(const std::string &library);

/// Source name of a kernel symbol: the original name of a renamed kernel, otherwise name.
/// @param name Kernel symbol.
/// @return The name the program and the reference runs use.
std::string source_name(const std::string &name);

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
