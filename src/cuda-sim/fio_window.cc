// fio_window.cc: see fio_window.h.
#include "fio_window.h"

#include "host_window.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
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
  uint64_t addr;              ///< first address of the piece
  uint64_t visible_tick;      ///< logical time at which the piece becomes visible
  uint64_t seq;               ///< submission order, breaks ties at equal visible_tick
  std::vector<unsigned char> data;  ///< the piece's bytes
};

/// Settings read once from the environment.
struct Config {
  uint64_t tear_unit;    ///< bytes per independently-visible piece (YASAI_FIO_WINDOW_TEAR, default 1)
  uint64_t delay_max;    ///< largest extra delay of a piece in ticks (YASAI_FIO_WINDOW_DELAY, default 32)
  bool strict;           ///< true: only fence and completion flush; false: kernel end flushes too
  uint64_t seed;         ///< RNG seed (YASAI_FIO_WINDOW_SEED, default 1)
  std::string record_path;  ///< FIO transport record file (YASAI_FIO_WINDOW_RECORD), empty: none
};

/// @return A nonzero unsigned environment value, or the default.
uint64_t env_u64(const char *name, uint64_t dflt) {
  const char *v = getenv(name);
  if (!v || !*v) return dflt;
  char *e = nullptr;
  uint64_t r = strtoull(v, &e, 0);
  return r;
}

/// @return The settings (read once).
Config &config() {
  static Config *c = [] {
    auto *cfg = new Config;
    cfg->tear_unit = std::max<uint64_t>(1, env_u64("YASAI_FIO_WINDOW_TEAR", 1));
    cfg->delay_max = std::max<uint64_t>(1, env_u64("YASAI_FIO_WINDOW_DELAY", 32));
    cfg->strict = env_u64("YASAI_FIO_WINDOW_KERNEL_FLUSH", 0) == 0;
    cfg->seed = env_u64("YASAI_FIO_WINDOW_SEED", 1);
    const char *rp = getenv("YASAI_FIO_WINDOW_RECORD");
    cfg->record_path = rp ? rp : "";
    return cfg;
  }();
  return *c;
}

/// Registered windows by first address.
std::map<uint64_t, Window> &windows() {
  static auto *m = new std::map<uint64_t, Window>;
  return *m;
}

/// Pending tear pieces of this process, newest last.
std::vector<Pending> &pending() {
  static auto *v = new std::vector<Pending>;
  return *v;
}

/// Guards windows(), pending() and the clock.
std::mutex &mu() {
  static auto *m = new std::mutex;
  return *m;
}

/// Logical clock; every store and load advances it.
uint64_t g_tick = 0;
/// Submission counter of pending pieces.
uint64_t g_seq = 0;
/// Seeded source of per-piece delays.
std::mt19937_64 &rng() {
  static auto *r = new std::mt19937_64(config().seed);
  return *r;
}
/// Record file handle (opened once when a window asks to record).
FILE *g_record = nullptr;

/// Counts for YASAI_FIO_WINDOW_STATS (printed at exit): stores intercepted, tear pieces created,
/// pieces made visible, and fences.
uint64_t g_stat_stores = 0, g_stat_pieces = 0, g_stat_flushed = 0, g_stat_fences = 0;
/// Registers the atexit stats dump once when stats are on.
void stats_arm() {
  static bool armed = []{
    if (getenv("YASAI_FIO_WINDOW_STATS"))
      atexit([]{
        // The settings go on the same line, so a run record carries the tear unit and seed.
        Config &c = config();
        fprintf(stderr, "yasai fio_window stats: stores=%llu pieces=%llu flushed=%llu fences=%llu "
                "tear=%llu delay=%llu seed=%llu flush=%s\n",
                (unsigned long long)g_stat_stores, (unsigned long long)g_stat_pieces,
                (unsigned long long)g_stat_flushed, (unsigned long long)g_stat_fences,
                (unsigned long long)c.tear_unit, (unsigned long long)c.delay_max,
                (unsigned long long)c.seed, c.strict ? "strict" : "kernel_end");
      });
    return true;
  }();
  (void)armed;
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

/// Writes one transport record line when recording is on.
void record(const char *kind, uint64_t addr, size_t bytes) {
  Config &cfg = config();
  if (cfg.record_path.empty()) return;
  if (!g_record) g_record = fopen(cfg.record_path.c_str(), "a");
  if (!g_record) return;
  fprintf(g_record, "{\"rec\":\"%s\",\"addr\":%llu,\"bytes\":%zu,\"tick\":%llu}\n", kind,
          (unsigned long long)addr, bytes, (unsigned long long)g_tick);
  fflush(g_record);
}

/// Applies the due pending pieces (visible_tick <= now) to shared memory, in (tick, seq) order.
/// The caller holds mu().
/// @param now The logical time up to which pieces are made visible.
void drain_locked(uint64_t now) {
  auto &p = pending();
  if (p.empty()) return;
  std::stable_sort(p.begin(), p.end(), [](const Pending &a, const Pending &b) {
    return a.visible_tick != b.visible_tick ? a.visible_tick < b.visible_tick : a.seq < b.seq;
  });
  size_t i = 0;
  for (; i < p.size() && p[i].visible_tick <= now; ++i) {
    host_window::write(p[i].addr, p[i].data.size(), p[i].data.data());
    g_stat_flushed++;
  }
  p.erase(p.begin(), p.begin() + i);
}

}  // namespace

void add(uint64_t base, size_t bytes, bool record) {
  if (bytes == 0) return;
  std::lock_guard<std::mutex> lock(mu());
  windows()[base] = Window{bytes, record};
}

void remove(uint64_t base) {
  std::lock_guard<std::mutex> lock(mu());
  windows().erase(base);
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
  const unsigned char *src = static_cast<const unsigned char *>(in);
  g_tick++;
  stats_arm();
  g_stat_stores++;
  for (size_t off = 0; off < bytes; off += cfg.tear_unit) {
    g_stat_pieces++;
    const size_t n = std::min<uint64_t>(cfg.tear_unit, bytes - off);
    Pending piece;
    piece.addr = addr + off;
    piece.visible_tick = g_tick + 1 + (rng()() % cfg.delay_max);
    piece.seq = g_seq++;
    piece.data.assign(src + off, src + off + n);
    pending().push_back(std::move(piece));
  }
  if (w->second.record) record("store", addr, bytes);
  return true;
}

void before_load(uint64_t addr, size_t bytes) {
  (void)addr;
  (void)bytes;
  std::lock_guard<std::mutex> lock(mu());
  if (pending().empty() && windows().empty()) return;
  g_tick++;
  drain_locked(g_tick);
}

void overlay(uint64_t addr, size_t bytes, void *out) {
  std::lock_guard<std::mutex> lock(mu());
  auto &p = pending();
  if (p.empty()) return;
  unsigned char *dst = static_cast<unsigned char *>(out);
  // Apply the process's own pending pieces in submission order, so the issuing kernel reads its
  // own stores (program order), even before they are visible to other processes.
  std::vector<const Pending *> hits;
  for (const auto &piece : p)
    if (piece.addr < addr + bytes && addr < piece.addr + piece.data.size()) hits.push_back(&piece);
  std::stable_sort(hits.begin(), hits.end(),
                   [](const Pending *a, const Pending *b) { return a->seq < b->seq; });
  for (const Pending *piece : hits) {
    const uint64_t lo = std::max(addr, piece->addr);
    const uint64_t hi = std::min(addr + bytes, piece->addr + piece->data.size());
    memcpy(dst + (lo - addr), piece->data.data() + (lo - piece->addr), hi - lo);
  }
}

void fence(const char *reason) {
  std::lock_guard<std::mutex> lock(mu());
  g_stat_fences++;
  if (pending().empty()) {
    if (!windows().empty()) record(reason, 0, 0);
    return;
  }
  drain_locked(UINT64_MAX);
  record(reason, 0, 0);
}

void on_kernel_end() {
  if (config().strict) return;
  fence("kernel_end");
}

}  // namespace fio_window
