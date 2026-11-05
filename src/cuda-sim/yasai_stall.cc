// yasai_stall.cc: see yasai_stall.h.
#include "yasai_stall.h"

#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace yasai_stall {

thread_local uint64_t t_events = 0;
thread_local unsigned t_rmw_depth = 0;
thread_local read_table *t_reads = nullptr;

namespace {

/// Bytes of a load kept as they are; a longer load keeps a hash of its bytes.
constexpr size_t kRawBytes = 16;

/// Thread and instruction of the loads being recorded, and the index of the next load of that
/// instruction (an instruction may read several times, for example a vector load or an atomic).
thread_local const ptx_thread_info *t_thread = nullptr;
thread_local uint64_t t_pc = 0;
thread_local uint32_t t_seq = 0;

/// Reads an unsigned environment variable.
/// @param name Variable.
/// @param dflt Value when unset or empty.
/// @return The value.
uint64_t env_u64(const char *name, uint64_t dflt) {
  const char *v = getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  char *e = nullptr;
  const unsigned long long x = strtoull(v, &e, 10);
  if (e == v || *e != '\0') {
    fprintf(stderr, "yasai-stall: %s=%s is not a non-negative integer\n", name, v);
    exit(2);
  }
  return x;
}

/// FNV-1a hash of a byte range.
/// @param p Bytes.
/// @param n Count.
/// @return Hash.
uint64_t fnv(const void *p, size_t n) {
  const unsigned char *b = (const unsigned char *)p;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

}  // namespace

/// One load: where and what it read.
struct load_entry {
  uint64_t addr = 0;            ///< address
  uint32_t length = 0;          ///< bytes
  unsigned char raw[kRawBytes]; ///< the bytes (length <= kRawBytes)
  uint64_t hash = 0;            ///< hash of the bytes (length > kRawBytes)
};

/// Key of a load: thread, instruction and the load's index within the instruction.
struct load_key {
  const ptx_thread_info *thread;  ///< executing thread
  uint64_t pc;                    ///< instruction address
  uint32_t seq;                   ///< index of the load within the instruction
  bool operator==(const load_key &o) const {
    return thread == o.thread && pc == o.pc && seq == o.seq;
  }
};

/// Hash of a load_key.
struct load_key_hash {
  size_t operator()(const load_key &k) const {
    return size_t(fnv(&k.thread, sizeof k.thread) ^ (k.pc * 0x9e3779b97f4a7c15ull) ^
                  (uint64_t(k.seq) << 48));
  }
};

class read_table {
 public:
  /// Whether a load differed from the previous one of its key since tracking started.
  bool changed = false;
  /// Records a load; sets changed when its address or bytes differ from the previous one.
  /// @param k Key.
  /// @param addr Address.
  /// @param length Bytes.
  /// @param data The bytes read.
  void record(const load_key &k, uint64_t addr, size_t length, const void *data) {
    load_entry e;
    e.addr = addr;
    e.length = uint32_t(length);
    if (length <= kRawBytes)
      memcpy(e.raw, data, length);
    else
      e.hash = fnv(data, length);
    auto ins = m_last.emplace(k, e);
    if (ins.second) return;
    load_entry &o = ins.first->second;
    const bool same = o.addr == e.addr && o.length == e.length &&
                      (length <= kRawBytes ? memcmp(o.raw, e.raw, length) == 0 : o.hash == e.hash);
    if (!same) changed = true;
    o = e;
  }

 private:
  std::unordered_map<load_key, load_entry, load_key_hash> m_last;
};

void inst_begin(const ptx_thread_info *thread, uint64_t pc) {
  t_thread = thread;
  t_pc = pc;
  t_seq = 0;
}

void note_load(uint64_t addr, size_t length, const void *data) {
  read_table *r = t_reads;
  if (r == nullptr || r->changed || t_thread == nullptr) return;
  r->record(load_key{t_thread, t_pc, t_seq++}, addr, length, data);
}

uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

const settings &config() {
  static const settings s = [] {
    settings c;
    if (getenv("YASAI_HANG_ROUNDS") != nullptr) {
      fprintf(stderr,
              "yasai-stall: YASAI_HANG_ROUNDS is no longer read. Its total-rounds cap is "
              "YASAI_GROUP_ROUNDS_MAX; the stall judgement is YASAI_STALL_ROUNDS with "
              "YASAI_STALL_S (rounds without progress and wall seconds since the last progress). "
              "Unset YASAI_HANG_ROUNDS.\n");
      exit(2);
    }
    c.stall_rounds = env_u64("YASAI_STALL_ROUNDS", 0);
    if (c.stall_rounds != 0) {
      const char *v = getenv("YASAI_STALL_S");
      if (v == nullptr || *v == '\0') {
        fprintf(stderr,
                "yasai-stall: YASAI_STALL_ROUNDS is set but YASAI_STALL_S is not; both are "
                "required (YASAI_STALL_S=0 judges by rounds alone)\n");
        exit(2);
      }
      char *e = nullptr;
      c.stall_s = strtod(v, &e);
      if (e == v || *e != '\0' || c.stall_s < 0) {
        fprintf(stderr, "yasai-stall: YASAI_STALL_S=%s is not a non-negative number\n", v);
        exit(2);
      }
    }
    // A sampled kernel's group that waits for a skipped CTA never makes progress; 20,000,000
    // quiet rounds as before (it counted all rounds before).
    c.sample_rounds = env_u64("YASAI_CTA_SAMPLE_HANG_ROUNDS", 20000000ull);
    c.total_max = env_u64("YASAI_GROUP_ROUNDS_MAX", 0);
    c.stats = env_u64("YASAI_STALL_STATS", 0) != 0;
    // Read tracking starts after half of the smallest quiet limit, so that a group is judged
    // only after at least that many tracked rounds without a changed load.
    uint64_t lim = 0;
    if (c.stall_rounds) lim = c.stall_rounds;
    if (c.sample_rounds && (lim == 0 || c.sample_rounds < lim)) lim = c.sample_rounds;
    c.track_rounds = env_u64("YASAI_STALL_TRACK_ROUNDS", lim / 2);
    if (c.track_rounds == 0) c.track_rounds = 1;
    if (c.stall_rounds)
      fprintf(stderr,
              "yasai-stall: on: a CTA group stops the process after %llu rounds without progress "
              "and %.0f s since its last progress (read tracking after %llu quiet rounds)\n",
              (unsigned long long)c.stall_rounds, c.stall_s, (unsigned long long)c.track_rounds);
    else
      fprintf(stderr,
              "yasai-stall: off (YASAI_STALL_ROUNDS unset); CTA-sampled kernels stop after %llu "
              "rounds without progress\n",
              (unsigned long long)c.sample_rounds);
    return c;
  }();
  return s;
}

tracker::tracker() = default;
tracker::~tracker() = default;

void tracker::restart() {
  rounds = 0;
  quiet = 0;
  silent = 0;
  quiet_since = 0;
  reads.reset();
}

uint64_t begin_round(tracker &t) {
  t_reads = t.reads.get();
  t_thread = nullptr;
  return t_events;
}

double quiet_seconds(const tracker &t) {
  if (t.quiet == 0) return 0;
  return double(now_ns() - t.quiet_since) * 1e-9;
}

/// Closes a quiet stretch for the statistics.
/// @param t The group's state.
static void close_stretch(tracker &t) {
  if (t.quiet > t.max_quiet) t.max_quiet = t.quiet;
  // The clock is read only for stretches long enough to matter.
  if (t.quiet >= 1024) {
    const double s = quiet_seconds(t);
    if (s > t.max_quiet_s) t.max_quiet_s = s;
  }
}

verdict end_round(tracker &t, uint64_t events_before, bool sampled) {
  const settings &c = config();
  t_reads = nullptr;
  ++t.rounds;
  const bool events = t_events != events_before;
  const bool load_change = t.reads && t.reads->changed;
  if (events) {
    if (t.silent > t.max_silent) t.max_silent = t.silent;
    t.silent = 0;
  } else {
    ++t.silent;
  }
  if (events || load_change) {
    if (t.quiet) {
      if (c.stats) close_stretch(t);
      t.quiet = 0;
    }
    t.reads.reset();
  } else {
    if (t.quiet++ == 0) t.quiet_since = now_ns();
    if (t.quiet == c.track_rounds && !t.reads) t.reads.reset(new read_table);
  }
  if (c.total_max != 0 && t.rounds == c.total_max) return verdict::total_max;
  if (c.stall_rounds != 0 && t.quiet >= c.stall_rounds &&
      (t.quiet == c.stall_rounds || (t.quiet & 1023) == 0) && quiet_seconds(t) >= c.stall_s)
    return verdict::stall;
  if (sampled && c.sample_rounds != 0 && t.quiet == c.sample_rounds) return verdict::sample_stall;
  return verdict::none;
}

void end_group(tracker &t) {
  if (!config().stats) return;
  if (t.silent > t.max_silent) t.max_silent = t.silent;
  if (t.quiet) close_stretch(t);
}

}  // namespace yasai_stall
