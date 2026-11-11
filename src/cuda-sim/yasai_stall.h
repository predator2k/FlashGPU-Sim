// yasai_stall.h: stall judgement of CTA groups in functional mode (yasai fakeenv simulation
// backend).
//
// A round executes one instruction of every warp of a CTA group (a CTA, or the CTAs of a
// cluster). A round makes progress when, on the thread that runs it, one of these happens:
//   1. a thread of the group finishes;
//   2. a barrier releases a waiting warp (bar.sync and named barriers, barrier.cluster) or an
//      mbarrier completes a phase;
//   3. a store reaches the global or shared space (st, st.async, cp.async and TMA writes,
//      multimem.st, tcgen05.st), or an atomic or reduction changes the value in memory (a failed
//      compare-and-swap, an add of 0 or an exchange with the same value changes nothing);
//   4. while read tracking is on: a load of the global or shared space by the same thread at the
//      same instruction reads another address or another value than the previous time.
// Register arithmetic, branches, a load that repeats its address and value (a poll), nanosleep and
// fences are not progress.
//
// Classes 1 to 3 are counted by hooks that bump a thread-local counter. Class 4 needs a table of
// the last address and value per (thread, instruction, access), so it is kept only for a group
// that has been quiet for track_rounds() rounds; the first change turns it off again.
//
// The judgement: a group whose quiet rounds (consecutive rounds without progress) reach
// YASAI_STALL_ROUNDS and whose last progress lies at least YASAI_STALL_S wall seconds back has
// stalled; the process prints where each warp is and stops. Without YASAI_STALL_ROUNDS the
// judgement is off (one line says so). A CTA-sampled kernel uses the same quiet count for
// YASAI_CTA_SAMPLE_HANG_ROUNDS (a group that waits for a skipped CTA never makes progress), and
// YASAI_GROUP_ROUNDS_MAX keeps the former total-rounds cap of YASAI_HANG_ROUNDS (off by default).
// YASAI_HANG_ROUNDS itself is no longer read: a process that sets it stops at start with a line
// naming the replacements, so that no script keeps the old meaning unnoticed.
//
// YASAI_STALL_STATS=1 prints, per kernel, the longest quiet stretch of its groups in rounds and
// wall seconds and the longest stretch without class 1 to 3 events, to choose YASAI_STALL_ROUNDS
// and YASAI_STALL_S from measurements.
#ifndef YASAI_STALL_H
#define YASAI_STALL_H

#include <cstddef>
#include <cstdint>
#include <memory>

class ptx_thread_info;

namespace yasai_stall {

/// Progress events of classes 1 to 3 counted on the calling thread.
extern thread_local uint64_t t_events;
/// Depth of rmw_scope on the calling thread: stores inside it are counted by the atomic itself.
extern thread_local unsigned t_rmw_depth;

/// Counts one progress event (classes 1 and 2, and class 3 where the caller decides).
inline void note() { ++t_events; }
/// Counts a store to the global or shared space unless an atomic performs it.
inline void note_store() {
  if (t_rmw_depth == 0) ++t_events;
}
/// Counts an atomic or reduction that changed memory.
/// @param before Value in memory before the operation.
/// @param after Value written.
inline void note_rmw(uint64_t before, uint64_t after) {
  if (before != after) ++t_events;
}

/// Marks the store of an atomic or reduction, which note_rmw judges by its values.
struct rmw_scope {
  rmw_scope() { ++t_rmw_depth; }
  ~rmw_scope() { --t_rmw_depth; }
  rmw_scope(const rmw_scope &) = delete;
  rmw_scope &operator=(const rmw_scope &) = delete;
};

/// Table of the last load per (thread, instruction, access) of one group (class 4).
class read_table;

/// The read table of the group that the calling thread runs, or nullptr when read tracking is
/// off for it.
extern thread_local read_table *t_reads;

/// Starts the loads of one instruction (called before each instruction while t_reads is set).
/// @param thread Executing thread.
/// @param pc Instruction address.
void inst_begin(const ptx_thread_info *thread, uint64_t pc);
/// Records a load of the global or shared space while t_reads is set.
/// @param addr Address in its space.
/// @param length Bytes read.
/// @param data The bytes read.
void note_load(uint64_t addr, size_t length, const void *data);

/// Settings, read once from the environment.
struct settings {
  uint64_t stall_rounds = 0;   ///< YASAI_STALL_ROUNDS (0: judgement off)
  double stall_s = 0;          ///< YASAI_STALL_S
  uint64_t sample_rounds = 0;  ///< YASAI_CTA_SAMPLE_HANG_ROUNDS (quiet rounds; 0: off)
  uint64_t total_max = 0;      ///< YASAI_GROUP_ROUNDS_MAX (total rounds; 0: off)
  uint64_t track_rounds = 0;   ///< quiet rounds after which read tracking starts
  bool stats = false;          ///< YASAI_STALL_STATS
};
/// The settings. The first call checks the environment: YASAI_HANG_ROUNDS stops the process,
/// YASAI_STALL_ROUNDS without YASAI_STALL_S stops it, and one line states the judgement.
const settings &config();

/// Stall state of one CTA group (reset when the group takes its next CTA or cluster).
struct tracker {
  uint64_t rounds = 0;       ///< rounds executed by the current CTA or cluster
  uint64_t quiet = 0;        ///< consecutive rounds without progress
  uint64_t silent = 0;       ///< consecutive rounds without class 1 to 3 events
  uint64_t quiet_since = 0;  ///< monotonic ns of the last progress (valid while quiet > 0)
  std::unique_ptr<read_table> reads;  ///< read table while tracking is on
  uint64_t max_quiet = 0;    ///< longest quiet stretch (YASAI_STALL_STATS)
  double max_quiet_s = 0;    ///< its longest wall time in seconds
  uint64_t max_silent = 0;   ///< longest stretch without class 1 to 3 events
  tracker();
  ~tracker();
  /// Starts the next CTA or cluster; the maxima are kept.
  void restart();
};

/// What end_round found.
enum class verdict : uint8_t {
  none,         ///< go on
  stall,        ///< YASAI_STALL_ROUNDS and YASAI_STALL_S reached
  sample_stall, ///< YASAI_CTA_SAMPLE_HANG_ROUNDS reached by a CTA-sampled kernel
  total_max,    ///< YASAI_GROUP_ROUNDS_MAX reached
};

/// Before a round of the group on the calling thread.
/// @param t The group's state.
/// @return The event count to pass to end_round.
uint64_t begin_round(tracker &t);
/// After a round of the group that left a thread live.
/// @param t The group's state.
/// @param events_before begin_round's result.
/// @param sampled Whether the kernel is CTA-sampled.
/// @return The verdict.
verdict end_round(tracker &t, uint64_t events_before, bool sampled);
/// Closes the group's current quiet stretch for the statistics (the group finished).
/// @param t The group's state.
void end_group(tracker &t);
/// Wall seconds since the group's last progress (0 when it is not quiet).
/// @param t The group's state.
double quiet_seconds(const tracker &t);

/// Real monotonic clock in ns (coarse; not stretched by a time-dilation preload).
uint64_t now_ns();

}  // namespace yasai_stall

#endif  // YASAI_STALL_H
