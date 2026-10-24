// host_window.h: identity-mapped host memory for kernels (yasai fakeenv simulation backend).
//
// Pinned host memory that a program maps into the device address space (cudaHostAlloc with
// cudaHostAllocMapped, cudaMallocHost, cudaHostRegister) is registered here. Its device address
// equals its host address, the way unified virtual addressing works on a real GPU, and the global
// memory space reads and writes registered ranges directly in host memory instead of in the
// simulator's own storage. Kernels and host threads (and, through shared memory, other processes)
// therefore see each other's stores while the kernel runs, which communication libraries need
// (NCCL's proxy thread and its kernels exchange FIFO counters through such memory).
//
// Device memory that is shared with other processes (exported with cudaIpcGetMemHandle, created
// with cuMemCreate and mapped with cuMemMap, or opened from another process) is registered here
// too, as range_kind::device: the range is a POSIX shared memory mapping placed at the device
// address itself (inside the simulator's device heap), so it is identity mapped as well and has the
// same acquire/release visibility. A range of kind multicast is a virtual address range mapped to a
// multicast object (cuMulticastCreate); it has no host memory of its own, and its accesses are
// forwarded to the multicast hooks (shared_devmem), which reach the memory each device bound.
//
// Device heap addresses of the simulator start at GLOBAL_HEAP_START (0xC00000000) and grow
// upward; user-space host addresses on Linux x86-64 and aarch64 lie far above any realistic heap
// size, so the two address sets do not overlap in practice. whichspace() classifies both as global.
#ifndef HOST_WINDOW_H
#define HOST_WINDOW_H

#include <cstddef>
#include <cstdint>

namespace host_window {

/// What a registered range holds.
enum class range_kind : uint8_t {
  host,       ///< pinned host memory mapped for the device (device address == host address)
  device,     ///< shared device memory placed at its device address (identity mapped)
  multicast,  ///< multicast virtual addresses; accesses go to the multicast hooks
};

/// One registered range, as returned by lookup().
struct range_info {
  uint64_t base;       ///< first address of the range
  size_t bytes;        ///< length of the range in bytes
  range_kind kind;     ///< what the range holds
  void *mc;            ///< multicast object of a multicast range (nullptr otherwise)
  uint64_t mc_offset;  ///< offset in the multicast object of the range's first byte
};

/// Reads `bytes` bytes at offset `off` of multicast object `mc` into `out` (one device's copy).
typedef void (*mc_read_fn)(void *mc, uint64_t off, size_t bytes, void *out);
/// Writes `bytes` bytes from `in` at offset `off` of multicast object `mc` (to every device).
typedef void (*mc_write_fn)(void *mc, uint64_t off, size_t bytes, const void *in);

/// Registers [base, base + bytes) as memory that kernels access in place.
/// A range that overlaps an existing one replaces it.
/// @param base First host (and device) address of the range.
/// @param bytes Length of the range in bytes; 0 is ignored.
/// @param kind What the range holds (host memory by default).
/// @param mc Multicast object of a multicast range.
/// @param mc_offset Offset in the multicast object that `base` corresponds to.
void add(uint64_t base, size_t bytes, range_kind kind = range_kind::host, void *mc = nullptr,
         uint64_t mc_offset = 0);

/// Removes the range that starts at base, if any.
/// @param base First address of a range passed to add().
/// @return True when a range was removed.
bool remove(uint64_t base);

/// Whether [addr, addr + bytes) lies inside one registered range.
/// @param addr First address of the access.
/// @param bytes Length of the access in bytes.
/// @return True when the access must go to host memory.
bool contains(uint64_t addr, size_t bytes);

/// Finds the registered range that holds [addr, addr + bytes).
/// @param addr First address of the access.
/// @param bytes Length of the access in bytes.
/// @param out Receives the range when one is found.
/// @return True when one range holds the whole access.
bool lookup(uint64_t addr, size_t bytes, range_info *out);

/// Length of the range that starts at base.
/// @param base First address of a range passed to add().
/// @return Length in bytes, or 0 when base does not start a registered range.
size_t size_at(uint64_t base);

/// Installs the functions that serve accesses to multicast ranges.
/// @param rd Reads one device's copy.
/// @param wr Writes every device's copy.
void set_multicast_hooks(mc_read_fn rd, mc_write_fn wr);

/// Reads host memory for a kernel load; aligned 1, 2, 4 and 8 byte accesses are single atomic
/// (acquire) loads so that spinning on a flag observes another thread's store. Aligned 16-byte
/// accesses are single atomic loads too: MOVDQA on Intel and AMD processors with AVX, otherwise
/// lock cmpxchg16b (which writes, so it commits a never-written page).
/// @param addr Host address.
/// @param bytes Length in bytes.
/// @param out Destination buffer.
void read(uint64_t addr, size_t bytes, void *out);

/// Writes host memory for a kernel store; aligned 1, 2, 4 and 8 byte accesses are single atomic
/// (release) stores, and a release fence follows every store so earlier stores of the kernel are
/// visible no later than this one.
/// @param addr Host address.
/// @param bytes Length in bytes.
/// @param in Source buffer.
void write(uint64_t addr, size_t bytes, const void *in);

/// Serves a load when a registered range holds it (identity ranges through read(), multicast
/// ranges through the multicast read hook).
/// @param addr First address of the access.
/// @param bytes Length in bytes.
/// @param out Destination buffer.
/// @return True when the access was served; false when no range holds it.
bool try_read(uint64_t addr, size_t bytes, void *out);

/// Serves a store when a registered range holds it (identity ranges through write(), multicast
/// ranges through the multicast write hook, which stores to every device). With `weak` set, a
/// store into device memory whose bytes and target are both zero is skipped (page by page for
/// writes longer than 16 bytes): the memory already holds the value, and the never-written pages
/// of cudaMalloc memory stay uncommitted. Callers set `weak` only for a weak st (no .release,
/// .relaxed, .volatile or .mmio) and for memset and copies (copy_zero_skip_scope), never for
/// atomics, reductions or ordered stores. YASAI_DEVMEM_ZERO_ELIDE=0 or YASAI_TRACE_DIR turns the
/// skip off.
/// @param addr First address of the access.
/// @param bytes Length in bytes.
/// @param in Source buffer.
/// @param weak The store may be skipped when it stores zeros over zeros.
/// @return True when the access was served; false when no range holds it.
bool try_write(uint64_t addr, size_t bytes, const void *in, bool weak = false);

/// Marks this thread's stores as memset or copy stores while the object lives, so that
/// memory_space_impl::write passes `weak` to try_write for stores without an instruction.
struct copy_zero_skip_scope {
  /// Enters the scope.
  copy_zero_skip_scope();
  /// Leaves the scope.
  ~copy_zero_skip_scope();
};

/// Whether this thread is inside a copy_zero_skip_scope.
/// @return True inside the scope.
bool in_copy_zero_skip_scope();

/// Number of stores (or 4 KiB pieces of long writes) skipped as zero over zero so far.
/// @return The count.
uint64_t zero_skipped_count();

/// Atomic compare-and-swap of a naturally aligned 2, 4 or 8 byte word in host memory.
/// @param addr Host address of the word.
/// @param bytes Size of the word (2, 4 or 8).
/// @param expected In: the value the word must hold; out: the value it held.
/// @param desired Value stored when the word holds *expected.
/// @return True when the swap happened.
bool compare_exchange(uint64_t addr, size_t bytes, uint64_t *expected, uint64_t desired);

}  // namespace host_window

#endif  // HOST_WINDOW_H
