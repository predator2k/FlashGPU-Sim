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
// Device heap addresses of the simulator start at GLOBAL_HEAP_START (0xC00000000) and grow
// upward; user-space host addresses on Linux x86-64 and aarch64 lie far above any realistic heap
// size, so the two address sets do not overlap in practice. whichspace() classifies both as global.
#ifndef HOST_WINDOW_H
#define HOST_WINDOW_H

#include <cstddef>
#include <cstdint>

namespace host_window {

/// Registers [base, base + bytes) as host memory that kernels access in place.
/// A range that overlaps an existing one replaces it.
/// @param base First host (and device) address of the range.
/// @param bytes Length of the range in bytes; 0 is ignored.
void add(uint64_t base, size_t bytes);

/// Removes the range that starts at base, if any.
/// @param base First address of a range passed to add().
/// @return True when a range was removed.
bool remove(uint64_t base);

/// Whether [addr, addr + bytes) lies inside one registered range.
/// @param addr First address of the access.
/// @param bytes Length of the access in bytes.
/// @return True when the access must go to host memory.
bool contains(uint64_t addr, size_t bytes);

/// Length of the range that starts at base.
/// @param base First address of a range passed to add().
/// @return Length in bytes, or 0 when base does not start a registered range.
size_t size_at(uint64_t base);

/// Reads host memory for a kernel load; aligned 1, 2, 4 and 8 byte accesses are single atomic
/// (relaxed) loads so that spinning on a flag observes another thread's store.
/// @param addr Host address.
/// @param bytes Length in bytes.
/// @param out Destination buffer.
void read(uint64_t addr, size_t bytes, void *out);

/// Writes host memory for a kernel store; aligned 1, 2, 4 and 8 byte accesses are single atomic
/// (relaxed) stores, and a release fence follows every store so earlier stores of the kernel are
/// visible no later than this one.
/// @param addr Host address.
/// @param bytes Length in bytes.
/// @param in Source buffer.
void write(uint64_t addr, size_t bytes, const void *in);

}  // namespace host_window

#endif  // HOST_WINDOW_H
