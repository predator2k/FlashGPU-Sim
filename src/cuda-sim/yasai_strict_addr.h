// yasai_strict_addr.h: illegal-address checks of the global space (yasai fakeenv simulation backend).
//
// An access of the global space that no host_window range holds goes to the simulator's own
// storage. Storage legitimately holds module variables (the static region below the high-water of
// the global allocator), heap allocations kept in storage (YASAI_DEVMEM_HOST=0, or an allocation
// that could not be placed as host memory) and a few allocations the simulator makes itself (CUDA
// arrays, the CDP parameter buffer, the cooperative grid workspace). Any other address has no
// allocation or mapping: on a GPU the access faults (cudaErrorIllegalAddress), here it used to
// read zeros and drop stores silently. check() is called for every part of a global-space access
// that goes to storage and decides:
//   * valid: one of the classes above holds the whole part;
//   * past the requested size but inside the mapping granularity (kGranularity, 2 MiB) of the
//     nearest heap allocation below it: a GPU maps whole pages, so such an access does not fault;
//     reported at warn level only, once per (kernel or copy, page);
//   * illegal: reported with the access, the nearest ranges and the nearest freed range; with
//     YASAI_STRICT_ADDR=1 (default) the process aborts, with warn it goes on, with 0 nothing is
//     checked.
// The report names the kernel and instruction of the access (inst_scope, set per executed
// instruction) or the copy or memset (copy_scope).
// The lookups take shared_devmem's lock. A simulator path that reads or writes the global space
// while it holds that lock either declares the range it owns (own_scope: shared_devmem's IPC
// export copying a storage allocation into a segment) or stops with an internal-error line, rather
// than waiting for the lock its own thread holds.
#ifndef YASAI_STRICT_ADDR_H
#define YASAI_STRICT_ADDR_H

#include <cstddef>
#include <cstdint>

class ptx_thread_info;
class ptx_instruction;

namespace yasai_strict_addr {

/// Mapping granularity of device memory on the modelled GPUs (2 MiB, the minimum granularity
/// cuMemGetAllocationGranularity reports on them).
constexpr uint64_t kGranularity = uint64_t(2) << 20;

/// What check() does.
enum class mode_kind : uint8_t {
  off,    ///< YASAI_STRICT_ADDR=0: no check
  warn,   ///< YASAI_STRICT_ADDR=warn: print and go on
  fatal,  ///< YASAI_STRICT_ADDR=1 or unset: print and abort on an illegal address
};

/// The mode, from YASAI_STRICT_ADDR (read once).
/// @return The mode.
mode_kind mode();

/// Records the end of the static region after the global allocator placed a variable.
/// @param end First address after the variable.
void note_static_end(uint64_t end);

/// Records an allocation that the simulator makes in storage outside the allocation registry of
/// shared_devmem (CUDA arrays, CDP parameter buffers, cooperative grid workspaces).
/// @param base First address.
/// @param size Length in bytes.
void note_internal(uint64_t base, size_t size);

/// Records a range that was freed or unmapped (kept in a small ring for the reports) and
/// invalidates the per-thread caches of validated ranges.
/// @param base First address.
/// @param size Length in bytes.
/// @param what The call that freed it.
void note_freed(uint64_t base, size_t size, const char *what);

/// Checks a part of a global-space access that no host_window range holds.
/// @param addr First address.
/// @param bytes Length in bytes.
/// @param store Whether the access writes.
void check(uint64_t addr, size_t bytes, bool store);

/// Names the copy or memset that the calling thread performs while the object lives.
struct copy_scope {
  /// @param what Description, a string literal (for example "cudaMemcpy D2D").
  explicit copy_scope(const char *what);
  ~copy_scope();
  copy_scope(const copy_scope &) = delete;
  copy_scope &operator=(const copy_scope &) = delete;
  const char *m_prev;  ///< enclosing description
};

/// Declares [lo, lo + n) valid for the calling thread's accesses while the object lives, so that
/// check() passes them without a registry lookup. Only for a caller that holds shared_devmem's lock
/// and accesses an allocation it holds there.
struct own_scope {
  /// @param lo First address.
  /// @param n Length in bytes.
  own_scope(uint64_t lo, size_t n);
  ~own_scope();
  own_scope(const own_scope &) = delete;
  own_scope &operator=(const own_scope &) = delete;
  uint64_t m_prev_lo;  ///< enclosing range's first address
  uint64_t m_prev_hi;  ///< enclosing range's end
};

/// Names the thread and instruction that the calling thread executes while the object lives.
struct inst_scope {
  /// @param t Executing thread.
  /// @param pI Executed instruction.
  inst_scope(const ptx_thread_info *t, const ptx_instruction *pI);
  ~inst_scope();
  inst_scope(const inst_scope &) = delete;
  inst_scope &operator=(const inst_scope &) = delete;
  const ptx_thread_info *m_prev_t;   ///< enclosing thread
  const ptx_instruction *m_prev_pI;  ///< enclosing instruction
};

}  // namespace yasai_strict_addr

#endif
