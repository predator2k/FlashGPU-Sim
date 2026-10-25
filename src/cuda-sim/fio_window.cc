// fio_window.cc: see fio_window.h.
#include "fio_window.h"

#include "host_window.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fio_window {
namespace {

/// One registered window.
struct Window {
  size_t bytes;  ///< length of the window
  bool record;   ///< whether to write transport records for it
};

/// One tear piece of a buffered store, waiting to become visible to other processes.
struct Pending {
  uint64_t addr;      ///< first address of the piece
  uint64_t deadline;  ///< window-clock time at which the piece becomes visible
  uint64_t seq;       ///< submission order, breaks ties at equal deadlines
  std::vector<unsigned char> data;  ///< the piece's bytes
};

/// Delay distributions of YASAI_FIO_WINDOW_DELAY_DIST.
enum class delay_dist {
  uniform,  ///< 1..D, every value equally likely
  pareto,   ///< heavy-tailed: floor(1 / u^(1/alpha)) for u uniform in (0, 1], capped at D
};

/// Settings read once from the environment.
struct Config {
  uint64_t tear_unit;    ///< bytes per independently-visible piece (YASAI_FIO_WINDOW_TEAR, default 1)
  uint64_t delay_max;    ///< largest delay of a piece in clock ticks (YASAI_FIO_WINDOW_DELAY, default 32)
  delay_dist dist;       ///< delay distribution (YASAI_FIO_WINDOW_DELAY_DIST, default uniform)
  double alpha;          ///< pareto shape (YASAI_FIO_WINDOW_DELAY_ALPHA, default 1.5)
  bool strict;           ///< false: every kernel end delivers all (YASAI_FIO_WINDOW_KERNEL_FLUSH=1)
  uint64_t seed;         ///< RNG seed (YASAI_FIO_WINDOW_SEED, default 1)
  uint64_t poll_us;      ///< drainer poll interval in microseconds (YASAI_FIO_WINDOW_POLL_US, default 20)
  uint64_t idle_step_us; ///< pause between idle clock steps (YASAI_FIO_WINDOW_IDLE_STEP_US, default 20)
  /// Wall time without any clock tick after which a busy process takes one clock step itself
  /// (YASAI_FIO_WINDOW_STALL_US, default 2000; 0 disables). A consumer that polls its own memory
  /// for a flag a peer writes through a window does not touch a window, so without this the clock
  /// stands still while kernels spin and a pending write never lands, against the eventual
  /// delivery the model promises.
  uint64_t stall_us;
  uint64_t max_pending_bytes;  ///< cap on pending memory (YASAI_FIO_WINDOW_MAX_PENDING_MB, default 256)
  bool cap_fatal;        ///< abort instead of delivering early at the cap (YASAI_FIO_WINDOW_CAP_FATAL=1)
  /// Keep the write order per byte (YASAI_FIO_WINDOW_SAME_ADDR_ORDER=1). Default 0, the FIO base
  /// specification: no ordering even between two posted writes from one source to one address
  /// (fio-spec part V 3.4, default-unordered-ordering-model). 1 is a hypothetical profile.
  bool same_addr_order;
  std::string clock_path;   ///< shared clock file (YASAI_FIO_WINDOW_CLOCK)
  std::string record_path;  ///< FIO transport record file (YASAI_FIO_WINDOW_RECORD), empty: none
};

/// @return An unsigned environment value, or the default when unset or empty.
uint64_t env_u64(const char *name, uint64_t dflt) {
  const char *v = getenv(name);
  if (!v || !*v) return dflt;
  return strtoull(v, nullptr, 0);
}

/// @return A string environment value, or the default when unset or empty.
std::string env_str(const char *name, const char *dflt) {
  const char *v = getenv(name);
  return (v && *v) ? std::string(v) : std::string(dflt);
}

/// @return The settings (read once).
Config &config() {
  static Config *c = [] {
    auto *cfg = new Config;
    cfg->tear_unit = std::max<uint64_t>(1, env_u64("YASAI_FIO_WINDOW_TEAR", 1));
    cfg->delay_max = std::max<uint64_t>(1, env_u64("YASAI_FIO_WINDOW_DELAY", 32));
    cfg->dist = env_str("YASAI_FIO_WINDOW_DELAY_DIST", "uniform") == "pareto" ? delay_dist::pareto
                                                                             : delay_dist::uniform;
    cfg->alpha = atof(env_str("YASAI_FIO_WINDOW_DELAY_ALPHA", "1.5").c_str());
    if (!(cfg->alpha > 0)) cfg->alpha = 1.5;
    cfg->strict = env_u64("YASAI_FIO_WINDOW_KERNEL_FLUSH", 0) == 0;
    cfg->seed = env_u64("YASAI_FIO_WINDOW_SEED", 1);
    cfg->poll_us = std::max<uint64_t>(1, env_u64("YASAI_FIO_WINDOW_POLL_US", 20));
    cfg->idle_step_us = env_u64("YASAI_FIO_WINDOW_IDLE_STEP_US", 20);
    cfg->stall_us = env_u64("YASAI_FIO_WINDOW_STALL_US", 2000);
    cfg->max_pending_bytes = std::max<uint64_t>(1, env_u64("YASAI_FIO_WINDOW_MAX_PENDING_MB", 256)) << 20;
    cfg->cap_fatal = env_u64("YASAI_FIO_WINDOW_CAP_FATAL", 0) != 0;
    cfg->same_addr_order = env_u64("YASAI_FIO_WINDOW_SAME_ADDR_ORDER", 0) != 0;
    cfg->clock_path = env_str("YASAI_FIO_WINDOW_CLOCK", "/dev/shm/yasai-fio-window-clock");
    cfg->record_path = env_str("YASAI_FIO_WINDOW_RECORD", "");
    return cfg;
  }();
  return *c;
}

/// The window clock: one 64-bit counter in a shared mapping of config().clock_path, so every
/// process sharing the windows advances and reads the same time. Falls back to a process-local
/// counter (with a warning) when the file cannot be mapped.
struct Clock {
  std::atomic<uint64_t> *now = nullptr;  ///< the counter
  bool shared = false;                   ///< whether the counter is the shared one
};

/// @return The clock (mapped once).
Clock &clock() {
  static Clock *c = [] {
    auto *ck = new Clock;
    const std::string &path = config().clock_path;
    const int fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd >= 0) {
      struct stat st;
      if (fstat(fd, &st) == 0 && st.st_size < 4096 && ftruncate(fd, 4096) != 0) {
        // keep going: mmap below fails if the file is still too short
      }
      void *p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      close(fd);
      if (p != MAP_FAILED) {
        ck->now = static_cast<std::atomic<uint64_t> *>(p);
        ck->shared = true;
      }
    }
    if (!ck->now) {
      fprintf(stderr, "yasai fio_window: cannot map the shared clock %s; using a process-local "
                      "clock (delivery then depends on this process's own window accesses)\n",
              path.c_str());
      ck->now = new std::atomic<uint64_t>(0);
    }
    return ck;
  }();
  return *c;
}

/// Advances the window clock by one tick.
/// @return The new time.
uint64_t clock_tick() { return clock().now->fetch_add(1, std::memory_order_acq_rel) + 1; }

/// @return The current window-clock time.
uint64_t clock_now() { return clock().now->load(std::memory_order_acquire); }

/// Raises the window clock to at least t (it never goes back).
/// @param t Target time.
void clock_advance_to(uint64_t t) {
  uint64_t cur = clock_now();
  while (cur < t && !clock().now->compare_exchange_weak(cur, t, std::memory_order_acq_rel)) {
  }
}

/// Orders pending pieces by (deadline, seq): the order in which they become visible.
struct by_deadline {
  bool operator()(const Pending *a, const Pending *b) const {
    return a->deadline != b->deadline ? a->deadline < b->deadline : a->seq < b->seq;
  }
};

/// Pending pieces of this process: owned in by_seq (submission order, keyed by seq), indexed by
/// deadline for delivery and by address for the issuing process's own reads.
struct PendingSet {
  std::map<uint64_t, Pending> by_seq;                     ///< seq -> piece (owner)
  std::set<const Pending *, by_deadline> by_time;         ///< delivery order
  std::multimap<uint64_t, const Pending *> by_addr;       ///< first address -> piece
  uint64_t bytes = 0;     ///< accounted memory of the pending pieces
  size_t max_piece = 0;   ///< largest piece length seen (bounds the by_addr search)
};

/// @return This process's pending pieces.
PendingSet &pending() {
  static auto *p = new PendingSet;
  return *p;
}

/// Accounted memory of one piece: the bytes plus the three index entries and allocator overhead.
/// @param n Piece length.
/// @return Bytes charged against the cap.
uint64_t piece_cost(size_t n) { return 160 + n; }

/// Registered windows by first address.
std::map<uint64_t, Window> &windows() {
  static auto *m = new std::map<uint64_t, Window>;
  return *m;
}

/// Guards windows(), pending(), the RNG, the counters and the drainer state.
std::mutex &mu() {
  static auto *m = new std::mutex;
  return *m;
}

/// Wakes the drainer when the first piece arrives or at exit.
std::condition_variable &drain_cv() {
  static auto *cv = new std::condition_variable;
  return *cv;
}

/// Submission counter of pending pieces.
uint64_t g_seq = 0;
/// Kernels of this process running now (on_kernel_start / on_kernel_end); 0 means the GPU is idle.
int g_running_kernels = 0;
/// Set at exit: the drainer stops, the exit handler delivers the rest.
bool g_exiting = false;
/// Whether the drainer thread and the exit handler are set up.
bool g_drainer_started = false;

/// @return The seeded source of per-piece delays.
std::mt19937_64 &rng() {
  static auto *r = new std::mt19937_64(config().seed);
  return *r;
}

/// Draws one piece delay from the configured distribution (caller holds mu()).
/// @return Delay in clock ticks, 1..D.
uint64_t draw_delay() {
  Config &cfg = config();
  if (cfg.dist == delay_dist::uniform) return 1 + rng()() % cfg.delay_max;
  // u in (0, 1]: 53 random bits, never 0.
  const double u = (double)((rng()() >> 11) + 1) * (1.0 / 9007199254740992.0);
  const double x = std::floor(std::pow(u, -1.0 / cfg.alpha));
  if (!(x < (double)cfg.delay_max)) return cfg.delay_max;
  return std::max<uint64_t>(1, (uint64_t)x);
}

/// Record file handle (opened once when a window asks to record).
FILE *g_record = nullptr;

/// Counts for YASAI_FIO_WINDOW_STATS (printed at exit): stores intercepted, tear pieces created,
/// pieces made visible, fences, idle clock steps, pieces delivered early at the pending cap, and
/// the largest pending memory.
uint64_t g_stat_stores = 0, g_stat_pieces = 0, g_stat_flushed = 0, g_stat_fences = 0;
uint64_t g_stat_idle_steps = 0, g_stat_cap_forced = 0, g_stat_peak_bytes = 0;
/// Clock steps taken while kernels ran but the clock stood still (Config::stall_us).
uint64_t g_stat_stall_steps = 0;

/// Writes one transport record line when recording is on.
void record(const char *kind, uint64_t addr, size_t bytes) {
  Config &cfg = config();
  if (cfg.record_path.empty()) return;
  if (!g_record) g_record = fopen(cfg.record_path.c_str(), "a");
  if (!g_record) return;
  fprintf(g_record, "{\"rec\":\"%s\",\"addr\":%llu,\"bytes\":%zu,\"tick\":%llu}\n", kind,
          (unsigned long long)addr, bytes, (unsigned long long)clock_now());
  fflush(g_record);
}

/// Makes the first piece in delivery order visible and drops it (caller holds mu()).
void deliver_front_locked() {
  PendingSet &p = pending();
  const Pending *piece = *p.by_time.begin();
  host_window::write(piece->addr, piece->data.size(), piece->data.data());
  g_stat_flushed++;
  p.by_time.erase(p.by_time.begin());
  auto range = p.by_addr.equal_range(piece->addr);
  for (auto it = range.first; it != range.second; ++it)
    if (it->second == piece) {
      p.by_addr.erase(it);
      break;
    }
  p.bytes -= piece_cost(piece->data.size());
  p.by_seq.erase(piece->seq);
}

/// Makes the pending pieces due at time now visible, in deadline order (caller holds mu()).
/// @param now Window-clock time up to which pieces are visible.
void drain_locked(uint64_t now) {
  PendingSet &p = pending();
  while (!p.by_time.empty() && (*p.by_time.begin())->deadline <= now) deliver_front_locked();
}

/// Delivers every pending piece, advancing the clock to each deadline in turn (caller holds mu()).
void deliver_all_locked() {
  PendingSet &p = pending();
  while (!p.by_time.empty()) {
    clock_advance_to((*p.by_time.begin())->deadline);
    deliver_front_locked();
  }
}

/// Prints the stats line (the settings go on the same line, so a run record carries them).
void print_stats() {
  Config &c = config();
  fprintf(stderr, "yasai fio_window stats: stores=%llu pieces=%llu flushed=%llu fences=%llu "
          "idle_steps=%llu stall_steps=%llu cap_forced=%llu peak_pending_mb=%.1f tear=%llu delay=%llu dist=%s "
          "alpha=%.2f seed=%llu flush=%s clock=%s same_addr_order=%d\n",
          (unsigned long long)g_stat_stores, (unsigned long long)g_stat_pieces,
          (unsigned long long)g_stat_flushed, (unsigned long long)g_stat_fences,
          (unsigned long long)g_stat_idle_steps, (unsigned long long)g_stat_stall_steps,
          (unsigned long long)g_stat_cap_forced,
          g_stat_peak_bytes / 1048576.0, (unsigned long long)c.tear_unit,
          (unsigned long long)c.delay_max, c.dist == delay_dist::pareto ? "pareto" : "uniform",
          c.alpha, (unsigned long long)c.seed, c.strict ? "strict" : "kernel_end",
          clock().shared ? "shared" : "local", c.same_addr_order ? 1 : 0);
}

/// Exit handler: stops the drainer and delivers what is still pending, step by step (eventual
/// delivery: a process that exits has posted its writes).
void at_exit() {
  std::unique_lock<std::mutex> lock(mu());
  g_exiting = true;
  drain_cv().notify_all();
  PendingSet &p = pending();
  while (!p.by_time.empty()) {
    clock_advance_to((*p.by_time.begin())->deadline);
    drain_locked(clock_now());
    g_stat_idle_steps++;
    if (config().idle_step_us) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(config().idle_step_us));
      lock.lock();
    }
  }
  if (getenv("YASAI_FIO_WINDOW_STATS")) print_stats();
}

/// Drainer thread: delivers pieces whose deadline the shared clock has passed; when no kernel of
/// this process runs, or kernels run but the clock has not moved for Config::stall_us, advances
/// the clock to the next deadline itself, one deadline at a time.
void drainer() {
  Config &cfg = config();
  std::unique_lock<std::mutex> lock(mu());
  uint64_t seen_clock = ~0ull;
  auto seen_at = std::chrono::steady_clock::now();
  for (;;) {
    // Nothing pending: sleep until a store or the exit handler wakes the thread. Otherwise poll
    // the clock (busy GPU) or take the next idle step (idle GPU).
    if (pending().by_time.empty()) {
      drain_cv().wait(lock, [] { return g_exiting || !pending().by_time.empty(); });
    } else {
      const bool stepping = g_running_kernels == 0;
      drain_cv().wait_for(lock, std::chrono::microseconds(
                                    stepping ? std::max<uint64_t>(1, cfg.idle_step_us) : cfg.poll_us));
    }
    if (g_exiting) return;
    PendingSet &p = pending();
    if (p.by_time.empty()) continue;
    drain_locked(clock_now());
    const auto wall = std::chrono::steady_clock::now();
    if (clock_now() != seen_clock) seen_clock = clock_now(), seen_at = wall;
    const bool idle = g_running_kernels == 0;
    const bool stalled = !idle && cfg.stall_us != 0 &&
                         wall - seen_at >= std::chrono::microseconds(cfg.stall_us);
    if ((idle || stalled) && !p.by_time.empty()) {
      clock_advance_to((*p.by_time.begin())->deadline);
      drain_locked(clock_now());
      if (idle)
        g_stat_idle_steps++;
      else
        g_stat_stall_steps++;
      seen_clock = clock_now(), seen_at = wall;
    }
  }
}

/// Starts the drainer thread and the exit handler once (caller holds mu()).
void start_drainer_locked() {
  if (g_drainer_started) return;
  g_drainer_started = true;
  clock();
  atexit(at_exit);
  std::thread(drainer).detach();
}

/// @return The window holding [addr, addr + bytes), or nullptr (caller holds mu()).
const std::pair<const uint64_t, Window> *find_locked(uint64_t addr, size_t bytes) {
  auto &m = windows();
  auto it = m.upper_bound(addr);
  if (it == m.begin()) return nullptr;
  --it;
  if (addr >= it->first && addr + bytes <= it->first + it->second.bytes) return &*it;
  return nullptr;
}

}  // namespace

void add(uint64_t base, size_t bytes, bool record) {
  if (bytes == 0) return;
  std::lock_guard<std::mutex> lock(mu());
  windows()[base] = Window{bytes, record};
}

void remove(uint64_t base) {
  std::lock_guard<std::mutex> lock(mu());
  auto w = windows().find(base);
  if (w == windows().end()) return;
  // Deliver this window's pending pieces (deadline order) before the mapping goes away.
  PendingSet &p = pending();
  const uint64_t end = base + w->second.bytes;
  std::vector<const Pending *> in;
  for (const Pending *piece : p.by_time)
    if (piece->addr >= base && piece->addr < end) in.push_back(piece);
  for (const Pending *piece : in) {
    clock_advance_to(piece->deadline);
    host_window::write(piece->addr, piece->data.size(), piece->data.data());
    g_stat_flushed++;
    p.by_time.erase(piece);
    auto range = p.by_addr.equal_range(piece->addr);
    for (auto it = range.first; it != range.second; ++it)
      if (it->second == piece) {
        p.by_addr.erase(it);
        break;
      }
    p.bytes -= piece_cost(piece->data.size());
    p.by_seq.erase(piece->seq);
  }
  windows().erase(w);
}

bool any() {
  std::lock_guard<std::mutex> lock(mu());
  return !windows().empty();
}

bool on_store(uint64_t addr, size_t bytes, const void *in) {
  std::lock_guard<std::mutex> lock(mu());
  const auto *w = find_locked(addr, bytes);
  if (!w) return false;
  Config &cfg = config();
  start_drainer_locked();
  const unsigned char *src = static_cast<const unsigned char *>(in);
  const uint64_t now = clock_tick();
  g_stat_stores++;
  PendingSet &p = pending();
  for (size_t off = 0; off < bytes; off += cfg.tear_unit) {
    g_stat_pieces++;
    const size_t n = std::min<uint64_t>(cfg.tear_unit, bytes - off);
    // At the cap, the oldest pieces by deadline are delivered early (the clock advances to their
    // deadlines), so the pending memory never grows past the cap; with YASAI_FIO_WINDOW_CAP_FATAL
    // the process aborts instead.
    while (!p.by_time.empty() && p.bytes + piece_cost(n) > cfg.max_pending_bytes) {
      if (cfg.cap_fatal) {
        fprintf(stderr, "yasai fio_window: pending writes reached YASAI_FIO_WINDOW_MAX_PENDING_MB "
                        "(%llu MB); aborting (YASAI_FIO_WINDOW_CAP_FATAL)\n",
                (unsigned long long)(cfg.max_pending_bytes >> 20));
        abort();
      }
      if (g_stat_cap_forced == 0)
        fprintf(stderr, "yasai fio_window: pending writes reached YASAI_FIO_WINDOW_MAX_PENDING_MB "
                        "(%llu MB); delivering the oldest by deadline early (counted as cap_forced)\n",
                (unsigned long long)(cfg.max_pending_bytes >> 20));
      clock_advance_to((*p.by_time.begin())->deadline);
      deliver_front_locked();
      g_stat_cap_forced++;
    }
    Pending piece;
    piece.addr = addr + off;
    piece.deadline = now + draw_delay();
    // Per-byte write order (hypothetical profile, YASAI_FIO_WINDOW_SAME_ADDR_ORDER=1): a piece never
    // becomes visible before an earlier pending piece that covers any of its bytes, so a later write
    // to a location is never overwritten by an earlier one. The FIO base specification (default)
    // gives no such order: an earlier write can land after a later one to the same address.
    if (cfg.same_addr_order && !p.by_addr.empty()) {
      const uint64_t a = addr + off;
      const uint64_t lo_key = a >= p.max_piece ? a - p.max_piece + 1 : 0;
      for (auto it = p.by_addr.lower_bound(lo_key); it != p.by_addr.end() && it->first < a + n; ++it)
        if (it->second->addr + it->second->data.size() > a)
          piece.deadline = std::max(piece.deadline, it->second->deadline);
    }
    piece.seq = g_seq++;
    piece.data.assign(src + off, src + off + n);
    auto ins = p.by_seq.emplace(piece.seq, std::move(piece));
    const Pending *pp = &ins.first->second;
    p.by_time.insert(pp);
    p.by_addr.emplace(pp->addr, pp);
    p.bytes += piece_cost(n);
    p.max_piece = std::max(p.max_piece, n);
  }
  g_stat_peak_bytes = std::max(g_stat_peak_bytes, p.bytes);
  if (w->second.record) record("store", addr, bytes);
  drain_cv().notify_one();
  return true;
}

void before_load(uint64_t addr, size_t bytes) {
  std::lock_guard<std::mutex> lock(mu());
  if (!find_locked(addr, bytes)) return;
  const uint64_t now = clock_tick();
  drain_locked(now);
}

void overlay(uint64_t addr, size_t bytes, void *out) {
  std::lock_guard<std::mutex> lock(mu());
  PendingSet &p = pending();
  if (p.by_seq.empty()) return;
  unsigned char *dst = static_cast<unsigned char *>(out);
  // Apply the process's own pending pieces in submission order, so the issuing kernel reads its
  // own stores (program order), even before they are visible to other processes. A piece that
  // overlaps [addr, addr + bytes) starts at most max_piece - 1 bytes before addr.
  const uint64_t lo_key = addr >= p.max_piece ? addr - p.max_piece + 1 : 0;
  std::vector<const Pending *> hits;
  for (auto it = p.by_addr.lower_bound(lo_key); it != p.by_addr.end() && it->first < addr + bytes; ++it) {
    const Pending *piece = it->second;
    if (piece->addr + piece->data.size() > addr) hits.push_back(piece);
  }
  if (hits.empty()) return;
  std::sort(hits.begin(), hits.end(), [](const Pending *a, const Pending *b) { return a->seq < b->seq; });
  for (const Pending *piece : hits) {
    const uint64_t lo = std::max(addr, piece->addr);
    const uint64_t hi = std::min(addr + bytes, piece->addr + piece->data.size());
    memcpy(dst + (lo - addr), piece->data.data() + (lo - piece->addr), hi - lo);
  }
}

void fence(const char *reason) {
  std::lock_guard<std::mutex> lock(mu());
  g_stat_fences++;
  if (pending().by_time.empty()) {
    if (!windows().empty()) record(reason, 0, 0);
    return;
  }
  deliver_all_locked();
  record(reason, 0, 0);
}

void on_kernel_start() {
  std::lock_guard<std::mutex> lock(mu());
  g_running_kernels++;
}

void on_kernel_end() {
  {
    std::lock_guard<std::mutex> lock(mu());
    if (g_running_kernels > 0) g_running_kernels--;
    drain_cv().notify_one();
  }
  if (!config().strict) fence("kernel_end");
}

}  // namespace fio_window
