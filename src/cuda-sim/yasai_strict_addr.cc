// yasai_strict_addr.cc: see yasai_strict_addr.h.
#include "yasai_strict_addr.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <utility>

#include "ptx_ir.h"
#include "ptx_sim.h"
#include "shared_devmem.h"

namespace yasai_strict_addr {
namespace {

/// First address the global allocator hands out (symbol_table starts m_global_next there).
constexpr uint64_t kStaticLo = 0x100;
/// Freed ranges kept for the reports.
constexpr int kFreedRing = 64;

/// End of the static region (module variables and constants placed by the global allocator).
std::atomic<uint64_t> g_static_end{kStaticLo};
/// Bumped by every free; a per-thread cache is valid only for the generation it was filled in.
std::atomic<uint64_t> g_gen{1};

/// Simulator-internal storage allocations: first address to length.
std::map<uint64_t, size_t> &internals() {
  static auto *m = new std::map<uint64_t, size_t>;  // never destroyed: used during exit
  return *m;
}
/// Guards internals().
std::shared_mutex &internals_mu() {
  static auto *m = new std::shared_mutex;
  return *m;
}

/// One freed range.
struct freed_entry {
  uint64_t base = 0;         ///< first address
  size_t size = 0;           ///< length
  const char *what = "";     ///< the call that freed it
};
/// Ring of the last kFreedRing freed ranges.
freed_entry g_freed[kFreedRing];
/// Next slot of g_freed.
unsigned g_freed_next = 0;
/// Guards g_freed and g_freed_next.
std::mutex &freed_mu() {
  static auto *m = new std::mutex;
  return *m;
}

/// (context, page) pairs already reported at warn level.
std::set<std::pair<std::string, uint64_t>> &warned() {
  static auto *s = new std::set<std::pair<std::string, uint64_t>>;
  return *s;
}
/// Guards warned().
std::mutex &warned_mu() {
  static auto *m = new std::mutex;
  return *m;
}

/// The calling thread's copy or memset (copy_scope), or nullptr.
thread_local const char *t_copy = nullptr;
/// The calling thread's executing thread and instruction (inst_scope), or nullptr.
thread_local const ptx_thread_info *t_thread = nullptr;
/// See t_thread.
thread_local const ptx_instruction *t_inst = nullptr;
/// The calling thread's last validated storage range [t_lo, t_hi) and its generation.
thread_local uint64_t t_lo = 0, t_hi = 0, t_gen = 0;

/// Rounds v up to a multiple of g.
uint64_t round_up(uint64_t v, uint64_t g) { return (v + g - 1) / g * g; }

/// Description of the access for a report line.
/// @return Text.
std::string context() {
  if (t_copy) return std::string(t_copy);
  if (t_thread && t_inst) {
    char b[256];
    const dim3 c = t_thread->get_ctaid(), t = t_thread->get_tid();
    snprintf(b, sizeof(b), "kernel %s, pc 0x%llx, cta (%u,%u,%u) tid (%u,%u,%u): ",
             const_cast<ptx_thread_info *>(t_thread)->get_kernel().name().c_str(),
             (unsigned long long)t_inst->get_PC(), c.x, c.y, c.z, t.x, t.y, t.z);
    return std::string(b) + t_inst->get_source();
  }
  return std::string("a host access");
}

/// Key of the context for the once-per-page warnings (the kernel name or the copy kind).
/// @return Key.
std::string context_key() {
  if (t_copy) return std::string(t_copy);
  if (t_thread) return const_cast<ptx_thread_info *>(t_thread)->get_kernel().name();
  return std::string("host");
}

/// Text of the freed range nearest to addr (containing it, else the closest), or "none".
/// @param addr Address of the access.
/// @return Text.
std::string nearest_freed(uint64_t addr) {
  std::lock_guard<std::mutex> lk(freed_mu());
  const freed_entry *best = nullptr;
  uint64_t best_d = ~0ull;
  for (const freed_entry &f : g_freed) {
    if (f.size == 0) continue;
    const uint64_t d = addr < f.base ? f.base - addr : (addr < f.base + f.size ? 0 : addr - (f.base + f.size) + 1);
    if (d < best_d) {
      best_d = d;
      best = &f;
    }
  }
  if (!best) return std::string("none");
  char b[160];
  snprintf(b, sizeof(b), "[0x%llx, +%zu) by %s%s", (unsigned long long)best->base, best->size,
           best->what, best_d == 0 ? " (holds the address: use after free)" : "");
  return std::string(b);
}

/// Text of the nearest allocations or mappings below and above addr.
/// @param addr Address of the access.
/// @return Text.
std::string neighbors_text(uint64_t addr) {
  uint64_t bb = 0, ab = 0;
  size_t bs = 0, as = 0;
  bool bmap = false, amap = false;
  const int got = shared_devmem::neighbors(addr, &bb, &bs, &bmap, &ab, &as, &amap);
  char b[160];
  std::string s = "below ";
  if (got & 1) {
    snprintf(b, sizeof(b), "[0x%llx, +%zu) %s", (unsigned long long)bb, bs,
             bmap ? "mapping" : "allocation");
    s += b;
  } else {
    s += "none";
  }
  s += ", above ";
  if (got & 2) {
    snprintf(b, sizeof(b), "[0x%llx, +%zu) %s", (unsigned long long)ab, as,
             amap ? "mapping" : "allocation");
    s += b;
  } else {
    s += "none";
  }
  return s;
}

/// Reports an access past the requested size inside the mapping granularity (once per context
/// and page).
void report_grain(uint64_t addr, size_t bytes, bool store, uint64_t base, size_t size) {
  const uint64_t page = addr & ~(kGranularity - 1);
  {
    std::lock_guard<std::mutex> lk(warned_mu());
    if (!warned().insert(std::make_pair(context_key(), page)).second) return;
  }
  fprintf(stderr,
          "yasai-strictaddr: warn: address 0x%llx (+%zu bytes, %s) in %s lies past the %zu "
          "requested bytes of the allocation at 0x%llx but inside its %llu-byte mapping "
          "granularity (no fault on a GPU); it reaches the simulator's storage\n",
          (unsigned long long)addr, bytes, store ? "store" : "load", context().c_str(), size,
          (unsigned long long)base, (unsigned long long)kGranularity);
}

/// Reports an illegal address and aborts in fatal mode.
void report_illegal(uint64_t addr, size_t bytes, bool store) {
  const bool fatal = mode() == mode_kind::fatal;
  if (!fatal) {
    const uint64_t page = addr & ~(kGranularity - 1);
    std::lock_guard<std::mutex> lk(warned_mu());
    if (!warned().insert(std::make_pair("illegal:" + context_key(), page)).second) return;
  }
  fprintf(stderr,
          "yasai-strictaddr: %sillegal address 0x%llx (+%zu bytes, %s) in %s: no allocation or "
          "mapping holds it; %s; nearest freed range %s\n",
          fatal ? "" : "warn: ", (unsigned long long)addr, bytes, store ? "store" : "load",
          context().c_str(), neighbors_text(addr).c_str(), nearest_freed(addr).c_str());
  if (fatal) {
    fprintf(stderr, "yasai-strictaddr: aborting (YASAI_STRICT_ADDR=warn reports and goes on, "
                    "YASAI_STRICT_ADDR=0 turns the check off)\n");
    fflush(stderr);
    fflush(stdout);
    abort();
  }
}

}  // namespace

mode_kind mode() {
  static const mode_kind m = [] {
    const char *v = getenv("YASAI_STRICT_ADDR");
    if (!v || !v[0]) return mode_kind::fatal;
    if (strcmp(v, "0") == 0) return mode_kind::off;
    if (strcmp(v, "warn") == 0) return mode_kind::warn;
    if (strcmp(v, "1") == 0) return mode_kind::fatal;
    fprintf(stderr, "yasai-strictaddr: YASAI_STRICT_ADDR=%s is not 0, warn or 1\n", v);
    abort();
  }();
  return m;
}

void note_static_end(uint64_t end) {
  uint64_t cur = g_static_end.load(std::memory_order_relaxed);
  while (end > cur && !g_static_end.compare_exchange_weak(cur, end, std::memory_order_release))
  {
  }
}

void note_internal(uint64_t base, size_t size) {
  if (size == 0) return;
  std::unique_lock<std::shared_mutex> lk(internals_mu());
  internals()[base] = size;
}

void note_freed(uint64_t base, size_t size, const char *what) {
  {
    std::lock_guard<std::mutex> lk(freed_mu());
    g_freed[g_freed_next % kFreedRing] = freed_entry{base, size, what};
    g_freed_next++;
  }
  g_gen.fetch_add(1, std::memory_order_acq_rel);
}

void check(uint64_t addr, size_t bytes, bool store) {
  if (bytes == 0 || mode() == mode_kind::off) return;
  const uint64_t end = addr + bytes;
  // Module variables and constants: one comparison, no lock.
  if (addr >= kStaticLo && end <= g_static_end.load(std::memory_order_acquire)) return;
  const uint64_t gen = g_gen.load(std::memory_order_acquire);
  if (t_gen == gen && addr >= t_lo && end <= t_hi) return;
  // Heap allocations kept in storage, cuMem mappings and IPC imports (shared_devmem's registry).
  uint64_t b = 0;
  size_t s = 0;
  if (shared_devmem::range_of(addr, &b, &s)) {
    if (end <= b + s) {
      t_lo = b;
      t_hi = b + s;
      t_gen = gen;
      return;
    }
  } else {
    std::shared_lock<std::shared_mutex> lk(internals_mu());
    auto it = internals().upper_bound(addr);
    if (it != internals().begin()) {
      --it;
      if (addr >= it->first && end <= it->first + it->second) {
        t_lo = it->first;
        t_hi = it->first + it->second;
        t_gen = gen;
        return;
      }
    }
  }
  // Past the requested size of the nearest heap allocation below, inside its granularity (a
  // cuMem mapping or IPC import counts its own mapping size, so it has no such margin).
  uint64_t bb = 0, ab = 0;
  size_t bs = 0, as = 0;
  bool bmap = false, amap = false;
  if ((shared_devmem::neighbors(addr, &bb, &bs, &bmap, &ab, &as, &amap) & 1) && !bmap &&
      addr >= bb && end <= round_up(bb + bs, kGranularity)) {
    report_grain(addr, bytes, store, bb, bs);
    return;
  }
  report_illegal(addr, bytes, store);
}

copy_scope::copy_scope(const char *what) : m_prev(t_copy) { t_copy = what; }

copy_scope::~copy_scope() { t_copy = m_prev; }

inst_scope::inst_scope(const ptx_thread_info *t, const ptx_instruction *pI)
    : m_prev_t(t_thread), m_prev_pI(t_inst) {
  t_thread = t;
  t_inst = pI;
}

inst_scope::~inst_scope() {
  t_thread = m_prev_t;
  t_inst = m_prev_pI;
}

}  // namespace yasai_strict_addr
