// fio_window.h: unordered-visibility windows for the FIO communication library (yasai fakeenv).
//
// A FIO fabric gives no ordering between writes without a fence, and no cache coherence. A window
// registered here models that: a store into it does not reach the shared memory at once, but waits
// in a per-process pending buffer and becomes visible to other processes after a seeded delay, in a
// seeded order, torn at a configurable granularity. fence.*.sys and a completion MMIO make the
// process's earlier writes visible at once. The issuing process's own reads stay program-order
// consistent, because a read overlays the process's pending writes.
//
// host_window delegates to this module for a store or load that falls in a registered FIO window;
// fence_impl calls fence(). Draining writes real bytes through host_window::write, so a drained
// piece lands in the shared mapping the peer process reads. The model is a correctness tool, not a
// timing model: the clock is a logical op counter, not simulated time.
#ifndef YASAI_FIO_WINDOW_H
#define YASAI_FIO_WINDOW_H

#include <cstddef>
#include <cstdint>

namespace fio_window {

/// Registers [base, base + bytes) as an unordered FIO window.
/// @param base First address (device address, identity mapped like a host_window device range).
/// @param bytes Length in bytes.
/// @param record Whether stores, fences and completion MMIO to this window are written to the FIO
///        transport record.
void add(uint64_t base, size_t bytes, bool record);

/// Removes the window that starts at base.
/// @param base First address passed to add().
void remove(uint64_t base);

/// Whether any FIO window is registered (lets host_window skip this module otherwise).
/// @return True when at least one window is registered.
bool any();

/// Handles a store that may fall in a FIO window.
/// @param addr First address of the store.
/// @param bytes Length in bytes.
/// @param in Source bytes.
/// @return True when the store fell in a window and was buffered (the caller does not write it);
///         false when no window holds it (the caller writes it normally).
bool on_store(uint64_t addr, size_t bytes, const void *in);

/// Drains the pending writes that are due at the current logical time into shared memory, before a
/// load reads it. The caller then reads shared memory and calls overlay().
/// @param addr First address of the load (only writes overlapping it need draining).
/// @param bytes Length in bytes.
void before_load(uint64_t addr, size_t bytes);

/// Overlays the issuing process's own pending writes onto a load result, so the issuing kernel
/// reads its own stores in program order even before they are visible to other processes.
/// @param addr First address of the load.
/// @param bytes Length in bytes.
/// @param out Load result to overlay in place.
void overlay(uint64_t addr, size_t bytes, void *out);

/// Makes every pending write of this process visible at once (fence.*.sys, or a completion MMIO).
/// @param reason A short tag for the record ("fence" or "completion").
void fence(const char *reason);

/// Flushes pending writes if the flush mode makes a kernel boundary a flush point (kernel-end-flush
/// mode); a no-op in the default strict mode.
void on_kernel_end();

}  // namespace fio_window

#endif  // YASAI_FIO_WINDOW_H
