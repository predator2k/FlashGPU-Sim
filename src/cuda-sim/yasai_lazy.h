// yasai_lazy.h: lazy kernel execution in the functional mode (yasai fakeenv simulation backend).
//
// With YASAI_LAZY_EXEC=1 a kernel launch whose global memory accesses can be bounded is recorded
// at launch as before (its kernel record, identity, launch configuration and arguments), but its
// CTAs do not run at its turn in its stream. The simulation thread registers it as finished in
// stream order instead, so events, stream queries, stream waits and cudaEventElapsedTime see it
// finish where it would have finished, and keeps it pending. A pending kernel runs when memory it
// may write is observed, or when memory it may read or write is about to change:
//   * a device-to-host, device-to-device or host-to-device copy, a memset, a symbol copy, or a
//     host-executed fill (the hooks in gpgpu_t's copy and memset functions);
//   * a kernel that runs now (an eager kernel): the pending kernels that write what it reads or
//     writes, or read what it writes, run first; an eager kernel whose accesses are not bounded
//     runs every pending kernel first;
//   * cudaFree and the other calls that release device memory, and the calls that share device
//     memory with other processes (IPC handles, cuMem export, multicast bindings, RDMA pinning),
//     which run every pending kernel that touches the range, or every pending kernel;
//   * a host function in a stream (cudaLaunchHostFunc), which runs every pending kernel.
// The pending kernels selected by an observation run in launch order, together with the earlier
// pending kernels they depend on (transitively: an earlier kernel that writes what a selected one
// reads or writes, or reads what it writes). The others stay pending; at process exit the kernels
// still pending have never run. At most YASAI_LAZY_MAX_PENDING (default 20000) launches are kept
// pending (each keeps its parameter memory); beyond that the oldest one runs.
//
// Access sets. A function's PTX is analysed once: every global or generic memory access (ld, st,
// atom, red, cp.async, the TMA copies) is traced back to the kernel parameter words, parameter
// structures and global variables its address derives from. A launch's read and write sets are
// the allocations that those parameter words hold (shared_devmem::range_of). A kernel runs eagerly
// (never deferred) when
//   * an address derives from a 64-bit value loaded from memory (a pointer read from device
//     memory), from no parameter at all, or from a called function's parameter;
//   * a set touches memory that the host or another process accesses directly: mapped or pinned
//     host memory, managed memory, memory shared through IPC, cuMem export or multicast, or an
//     allocation the registry does not know;
//   * it can synchronize with other kernels or the host: a global atom or red, multimem, a
//     non-weak or volatile global access, membar or fence above CTA scope;
//   * a warp trace is written (YASAI_TRACE_DIR): the traces follow the launch order;
//   * it prints (vprintf), allocates (malloc), launches kernels (CDP), reads textures or surfaces;
//   * it is cooperative, has clusters, interleaves its CTAs (YASAI_CTA_INTERLEAVE), is a
//     host-executed fill, or its name matches YASAI_LAZY_EAGER_REGEX (std::regex_search,
//     ECMAScript), the force-eager list.
// Observation points are the copy and memset functions, kernel starts and the calls above, and by
// default the synchronizations: cudaDeviceSynchronize, cudaStreamSynchronize and
// cudaEventSynchronize (and cuCtxSynchronize, cuStreamSynchronize, cuEventSynchronize) run every
// pending kernel when they return (YASAI_LAZY_SYNC_FLUSH=1, the default; see sync_point). Device
// memory is identity-mapped host memory in this simulator, so a host thread may read a cudaMalloc
// buffer directly after a synchronization, and memory measurements and simulator counters read
// after one see the kernels' effects. YASAI_LAZY_SYNC_FLUSH=0 makes synchronizations no
// observation points (more kernels stay deferred; such direct reads then see the memory before
// the pending kernels). cudaEventQuery and cudaStreamQuery are never observation points. Either
// way a synchronization reports the completion that the stream order gives a pending kernel, and
// cudaEventElapsedTime counts it as finished there. A host thread or another process that spins
// on device memory without a synchronization can only rely on memory of the kinds above (mapped,
// pinned, managed, shared), which no deferred kernel touches.
//
// Records. With YASAI_FAKEENV_OUT, the kernel record of a deferred launch carries
// "exec_deferred":1 next to the fields of the host fill; when a deferred kernel runs, a record
// {"rec":"deferred_exec","corr":<the launch's corr>,"t":<ns>} follows. A launch with
// exec_deferred and no deferred_exec record was never simulated (launch-only). host_executed:1
// with time_source "none" stays the host fill's mark (no simulated instruction, no simulated
// time); a deferred launch never carries time_source, so the converter times it as any kernel.
// YASAI_LAZY_REPORT=1 prints one line per deferral and per flush; every process prints a summary
// line at exit ("yasai-lazy: summary ...").
#ifndef YASAI_LAZY_H
#define YASAI_LAZY_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class kernel_info_t;
class cuda_sim;

namespace yasai_lazy {

/// Address range [lo, hi).
struct Range {
  uint64_t lo;  ///< first address
  uint64_t hi;  ///< address after the last byte
};

/// Launch-time state of one kernel launch (kernel_info_t::yasai_lazy()).
struct LaunchInfo {
  bool deferred = false;      ///< execution deferred at launch
  bool bounded = false;       ///< reads and writes cover every global access of the kernel
  bool started = false;       ///< the simulation thread has handled the launch (start())
  std::vector<Range> reads;   ///< allocations the kernel may read (bounded only)
  std::vector<Range> writes;  ///< allocations the kernel may write (bounded only)
  unsigned corr = 0;          ///< correlation id of the launch's kernel record (0: none)
  std::string reason;         ///< why the launch runs eagerly (empty when deferred)
};

/// Whether lazy execution is on (YASAI_LAZY_EXEC=1, read once).
bool enabled();

/// Computes the access sets of launch k and decides whether its execution is deferred. Called
/// by the launching host thread after the grid's parameters are set, before the launch is pushed
/// to its stream; never defers when lazy execution is off.
/// @param k The launch.
/// @param corr Correlation id of the launch's kernel record (0 without a recorder).
/// @param host_fill The launch is a host-executed fill.
/// @return The state, also stored in k.
std::shared_ptr<LaunchInfo> prepare(kernel_info_t &k, unsigned corr, bool host_fill);

/// Simulation thread, when launch k reaches its turn. A deferred launch becomes pending (this
/// module then owns k and runs it, or deletes it at exit) and true is returned; the caller
/// registers it as finished. Otherwise the pending kernels that k's accesses depend on run first
/// and false is returned.
/// @param k The launch.
/// @param sim The functional simulator (runs the pending kernels).
bool start(kernel_info_t &k, cuda_sim *sim);

/// An access by the host side to device memory: reads [r, r + rn) and writes [w, w + wn). Runs
/// the pending kernels it depends on (on the simulation thread; another thread asks it and
/// waits). A zero length is no access.
void observe(uint64_t r, uint64_t rn, uint64_t w, uint64_t wn);

/// Runs every pending kernel (an access that cannot be bounded).
/// @param why Short cause for the report line.
void observe_all(const char *why);

/// A host synchronization returned (cudaDeviceSynchronize, cudaStreamSynchronize,
/// cudaEventSynchronize and their driver forms): with YASAI_LAZY_SYNC_FLUSH (default 1) every
/// pending kernel runs, so that a host thread that reads device memory directly after it (device
/// memory is identity-mapped host memory here), a memory measurement or a simulator counter sees
/// the kernels' effects; YASAI_LAZY_SYNC_FLUSH=0 makes synchronizations no observation points.
/// @param why The call, for the report line.
void sync_point(const char *why);

/// Marks the calling thread as the simulation thread (gpgpu_sim_thread_concurrent).
void set_sim_thread();

/// Whether a host thread waits for the simulation thread to run pending kernels.
bool requests_waiting();

/// Simulation thread: runs the pending kernels that waiting host threads asked for.
void serve_requests();

/// Prints the summary line (at exit; also callable at any time).
void print_summary();

}  // namespace yasai_lazy

#endif  // YASAI_LAZY_H
