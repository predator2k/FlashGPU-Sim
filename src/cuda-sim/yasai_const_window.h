// yasai_const_window.h: the constant space of PTX modules (yasai fakeenv simulation backend).
//
// On a GPU every module has its own constant segment, and ld.const resolves its address in the
// segment of the module that holds the running kernel. The fakeenv ld.const probes measured on a
// GB10 (driver 580.95.05, 2026-11-06; tools/fakeenv/coverage/inf-dp-attn-ep/agll in the yasai
// repository) give the rules this file implements:
//   * the value of a .const symbol is its offset in the module's segment; the symbols lie in
//     declaration order, each aligned to its .align (two 16-byte symbols at 0 and 0x10);
//   * the segments of the modules lie in one pool, each at the next 256-byte boundary after the
//     previous one (a module after one of 16 B, 320 B and 16 KiB starts 0x100, 0x200 and 0x4000
//     after it), and an unloaded module's space is reused by the next load (lowest address first);
//   * cvta.const x is the segment's global address plus x, cvta.to.const g is g minus it (64-bit
//     arithmetic for any value); a module without constant data has base 0;
//   * ld.const at an offset whose access ends past the module's data reads 0, without a fault;
//   * isspacep.const holds exactly inside [base, base + size) of the running kernel's module, and
//     isspacep.global is false there;
//   * the host's address of a constant (cuModuleGetGlobal, cudaGetSymbolAddress) is the segment's
//     global address plus the offset, and host writes there, ld.const and generic loads through
//     cvta.const all see the same bytes.
// The segments live in the window [kWindowBase, kWindowBase + kWindowSize) of the simulator's
// global storage, so generic and global accesses read and write them like any global memory and
// the three views above are one storage. Which module a later segment follows depends on the order
// in which modules are loaded, and the simulator loads lazily, so concrete addresses and which
// module a read past a segment reaches are not compared with a GPU.
// A module is a source PTX file: the per-kernel pieces that lazy loading parses from one file share
// the file's segment (yasai_const::module_scope names the file while a piece is parsed).
// Not modelled, and stopped with a "not modeled" line (also appended to YASAI_CVTA_REPORT): banked
// constants (.const[n]), .extern .const declarations, and a .const symbol that two source modules
// define under one name (the simulator keeps one symbol table for all modules). The simulator does
// not unload modules (cuModuleUnload does nothing), so the reuse of an unloaded segment never
// happens here.
#ifndef YASAI_CONST_WINDOW_H
#define YASAI_CONST_WINDOW_H

#include <atomic>
#include <string>

class symbol;
class ptx_thread_info;

namespace yasai_const {

/// First address of the constant window. It lies above the device heap (which grows from
/// GLOBAL_HEAP_START, 0xC00000000), above the static region and the local and shared generic
/// windows (all below GLOBAL_HEAP_START), and below the parameters' copies (0x600000000000,
/// kernel_info.cc), which grow upward; Linux places host mappings (host_window) far above.
constexpr unsigned long long kWindowBase = 0x5F0000000000ull;
/// Length of the constant window: 256 MiB, so the window is [0x5F0000000000, 0x5F0010000000) and
/// its end stays below the parameters' copies at 0x600000000000. 4,096 modules of the largest
/// constant segment (64 KiB); the rank logs of the coverage runs show at most 4 modules with
/// constant data per process (census of 2026-11-06). Exceeding it stops the process.
constexpr unsigned long long kWindowSize = 0x10000000ull;
/// Alignment of a segment's base, measured on the GB10.
constexpr unsigned long long kSegmentAlign = 256;

/// The constant segment of one source module.
struct segment {
  std::string key;  ///< source PTX file
  std::atomic<unsigned long long> base{0};  ///< global address; 0 before placement and without data
  unsigned long long size = 0;      ///< bytes of constant data: end of the last symbol
  unsigned long long reserved = 0;  ///< bytes placed: size rounded up to kSegmentAlign
  std::atomic<bool> placed{false};  ///< base is final
};

/// While alive, names the source module of the PTX that the current thread parses (a per-kernel
/// piece is named after its source file). The caller holds yasai_func::ptx_load_mutex.
class module_scope {
 public:
  /// @param key Source PTX file.
  explicit module_scope(const std::string &key);
  ~module_scope();
  module_scope(const module_scope &) = delete;
  module_scope &operator=(const module_scope &) = delete;

 private:
  std::string m_previous;
};

/// The segment of the module being parsed, created when first asked for.
/// @param parse_file The file the parser reads (gpgpu_context::g_filename), the module's name when
///        no module_scope is active.
/// @return The segment (never null).
segment *current(const char *parse_file);

/// Places a .const symbol in the module's segment and returns its offset.
/// @param seg The module's segment.
/// @param bytes Size of the symbol.
/// @param align Its alignment (its .align, else its element size).
/// @param name Symbol name, for the messages.
/// @return The offset.
unsigned long long add_symbol(segment *seg, unsigned long long bytes, unsigned align, const char *name);

/// Gives every segment created so far its global address (lowest free address first), and
/// declares the window to the illegal-address check once. Called after each parse, before the
/// constants are initialized. Stops the process when the window is full.
void place_pending();

/// The global address of the segment (0 for a module without constant data). Places it if needed.
/// @param seg The segment, or null for a function parsed outside any module.
/// @return The address.
unsigned long long base(segment *seg);

/// The host's address of a .const symbol: its segment's address plus its offset.
/// @param sym The symbol.
/// @return The address.
unsigned long long host_address(const symbol *sym);

/// The segment of the kernel the thread runs (null when it has none).
/// @param thread The thread.
/// @return The segment.
segment *of_thread(ptx_thread_info *thread);

/// Where ld.const at offset reads bytes bytes for the running kernel.
/// @param seg The kernel's segment (may be null).
/// @param offset The address operand of ld.const.
/// @param bytes Length of the whole access.
/// @param[out] addr The global address of the data when the result is true.
/// @return False when the access ends past the module's data (the load reads 0).
bool resolve_load(segment *seg, unsigned long long offset, unsigned long long bytes, unsigned long long *addr);

/// isspacep.const of a generic address for the running kernel's segment.
/// @param seg The kernel's segment (may be null).
/// @param addr The address.
/// @return Whether addr lies in [base, base + size).
bool contains(segment *seg, unsigned long long addr);

/// Stops the process for a constant form the model does not cover: prints a "not modeled" line and
/// appends it to the file YASAI_CVTA_REPORT names.
/// @param what Which form (banked, extern, defined twice).
/// @param name Symbol name.
/// @param detail Where it was found.
[[noreturn]] void not_modeled(const char *what, const char *name, const std::string &detail);

}  // namespace yasai_const

#endif
