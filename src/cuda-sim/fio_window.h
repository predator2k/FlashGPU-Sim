// fio_window.h: unordered-visibility windows for the FIO communication library (yasai fakeenv).
//
// A FIO fabric delivers every posted write eventually, but out of order, torn, and after an
// unbounded but finite delay; it gives no ordering between writes without a fence, and no cache
// coherence. A window registered here models that. A store into a window does not reach the shared
// memory at once: each tear piece (YASAI_FIO_WINDOW_TEAR bytes) waits in a per-process pending
// buffer with a seeded random deadline on the window clock, and becomes visible when the clock
// passes its deadline. Pieces become visible in deadline order, which keeps the reordering and
// tearing hazards. As in the FIO base specification, writes to the same address are not ordered
// either (an earlier write can land after a later one and overwrite it); the hypothetical profile
// YASAI_FIO_WINDOW_SAME_ADDR_ORDER=1 keeps the order of writes to one byte (a piece's deadline is at
// least that of every earlier pending piece covering its bytes).
//
// The window clock is global simulated time: a counter in a shared mapping
// (YASAI_FIO_WINDOW_CLOCK, default /dev/shm/yasai-fio-window-clock) that every store into and load
// from a window, in any process sharing it, advances by one. A drainer thread of each process with
// pending pieces makes them visible when the clock passes their deadlines, so delivery does not
// depend on the issuing process touching the window again. When the issuing process's GPU is idle
// (no kernel running, for example after a kernel end or during a host sync), the drainer advances
// the clock to the next deadline itself, one deadline at a time (YASAI_FIO_WINDOW_IDLE_STEP_US
// apart), instead of flushing everything at once; a consumer in another process can still see a
// flag before the data across a kernel boundary. The drainer also takes one such step when kernels
// run but the clock has not moved for YASAI_FIO_WINDOW_STALL_US (default 2000 us of wall time):
// a consumer that spins on its own memory for a flag that a peer writes through a window touches
// no window, so the clock would otherwise stand still and the write would never land.
//
// fence.*.sys and the completion MMIO block the issuing thread until all of its earlier stores are
// visible: they advance the clock to the last pending deadline and deliver every piece in deadline
// order. The kernel-end-flush mode (YASAI_FIO_WINDOW_KERNEL_FLUSH=1) also does that at every kernel
// end, to attribute a failure to a missing fence versus a cross-kernel race.
//
// Delays (YASAI_FIO_WINDOW_DELAY = D ticks at most) follow YASAI_FIO_WINDOW_DELAY_DIST: uniform
// (default, 1..D) or pareto (heavy-tailed, shape YASAI_FIO_WINDOW_DELAY_ALPHA, default 1.5,
// capped at D), seeded by YASAI_FIO_WINDOW_SEED. The issuing process's own reads stay program-order
// consistent, because a read overlays the process's pending writes.
//
// host_window delegates to this module for a store or load that falls in a registered FIO window;
// fence_impl calls fence(); the stream manager calls on_kernel_start() and on_kernel_end().
// Delivering writes real bytes through host_window::write, so a delivered piece lands in the
// shared mapping the peer process reads. The model is a correctness tool, not a timing model.
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

/// Removes the window that starts at base, first delivering this process's pending pieces in it
/// (in deadline order), because the mapping is about to go away.
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

/// Advances the window clock for a load that falls in a window, and delivers the pending writes
/// that are due, before the load reads shared memory. The caller then reads shared memory and
/// calls overlay().
/// @param addr First address of the load.
/// @param bytes Length in bytes.
void before_load(uint64_t addr, size_t bytes);

/// Overlays the issuing process's own pending writes onto a load result, so the issuing kernel
/// reads its own stores in program order even before they are visible to other processes.
/// @param addr First address of the load.
/// @param bytes Length in bytes.
/// @param out Load result to overlay in place.
void overlay(uint64_t addr, size_t bytes, void *out);

/// Blocks until every pending write of this process is visible (fence.*.sys, or a completion
/// MMIO): advances the clock to the last pending deadline and delivers in deadline order.
/// @param reason A short tag for the record ("fence" or "completion").
void fence(const char *reason);

/// Notes that a kernel of this process started running (the GPU is not idle).
void on_kernel_start();

/// Notes that a kernel of this process finished. In the kernel-end-flush mode it also delivers
/// every pending write, as fence() does; in the default mode the drainer delivers them step by
/// step once no kernel is running.
void on_kernel_end();

}  // namespace fio_window

#endif  // YASAI_FIO_WINDOW_H
