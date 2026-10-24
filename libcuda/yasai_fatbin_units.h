// yasai: translation units of a library's fat binaries (kernel-name units for yasai_module_scope).
//
// Every translation unit of a CUDA library or program contributes one fat binary to the ELF
// section .nv_fatbin and registers it with __cudaRegisterFatBinary; the fat binary holds one PTX
// entry per architecture. cuobjdump numbers the PTX entries of the section in order
// ("libfoo.<n>.sm_XX.ptx", n from 1), so the PTX files of one fat binary are consecutive. Two
// translation units of one library may define the same mangled kernel with different bodies
// (both instantiate a template of a struct that each defines); each registration must then bind
// to the kernel of its own fat binary. The unit of a PTX file or of a registration is
// "<library key>#<fat binary number>", the same for all architectures of one translation unit.
#ifndef YASAI_FATBIN_UNITS_H_INCLUDED
#define YASAI_FATBIN_UNITS_H_INCLUDED

#include <string>
#include <vector>

namespace yasai_fatbin {

/// Unit of the PTX file with cuobjdump number ptx_number extracted from the ELF file elf_path.
/// @param elf_path Library or program the PTX was extracted from.
/// @param ptx_number cuobjdump number of the PTX file (the <n> of "name.<n>.sm_XX.ptx").
/// @return "<library key>#<fat binary number>", or "" when the section cannot be read or the
///         number lies outside it.
std::string unit_of_ptx_number(const std::string &elf_path, unsigned ptx_number);

/// Unit of an extracted PTX file named "<name>.<n>.sm_XX.ptx". Without elf_path the ELF file is
/// found among the files noted with note_elf (and the running program) by the library key of
/// name; a library registered after start-up passes its own path, because its files may be
/// named after its source files.
/// @param ptx_file Path or base name of the extracted file.
/// @param elf_path ELF file the PTX was extracted from ("" to look it up).
/// @return The unit, or "" when the file name has no number or no ELF file matches.
std::string unit_of_ptx_file(const std::string &ptx_file, const std::string &elf_path = "");

/// Unit of the fat binary that a __cudaRegisterFatBinary call passed (the wrapper's data pointer
/// locates the library with dladdr and the fat binary inside its .nv_fatbin section).
/// @param fat_cubin The fat binary wrapper of the registration.
/// @return The unit, or "" when it cannot be located.
std::string unit_of_fatbin(const void *fat_cubin);

/// Settles the PTX files that `cuobjdump -xptx all` extracted from the ELF files into dir.
/// cuobjdump names a file after its source file, so files of two translation units compiled
/// from sources of one name (or of two libraries) overwrite each other: every name listed more
/// than once is extracted again from its own fat binary and stored as
/// "<library key>.<number>.<arch>.ptx". Every listed file is recorded with its (ELF file,
/// cuobjdump number), which unit_of_ptx_file then uses.
/// @param elf_paths ELF files, in extraction order.
/// @param names For each ELF file, the PTX file names of `cuobjdump -lptx`, in its order.
/// @param dir Directory the files were extracted into.
/// @param run Shell command runner (returns the exit status).
/// @return For each ELF file, its file names after renaming, in listing order.
std::vector<std::vector<std::string>> settle_listing(
    const std::vector<std::string> &elf_paths, const std::vector<std::vector<std::string>> &names,
    const std::string &dir, int (*run)(const char *));

/// Records an ELF file whose extracted PTX files unit_of_ptx_file may be asked about (the
/// PYTORCH_BIN libraries, libraries registered after start-up).
/// @param elf_path Library or program path.
void note_elf(const std::string &elf_path);

}  // namespace yasai_fatbin

#endif
