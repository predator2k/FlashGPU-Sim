// yasai_const_window.cc: see yasai_const_window.h.
#include "yasai_const_window.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../kernel_info.h"
#include "ptx_ir.h"
#include "ptx_sim.h"
#include "yasai_strict_addr.h"

namespace yasai_const {
namespace {

/// Guards the registry and the placement.
std::mutex &registry_mutex() {
  static auto *m = new std::mutex();
  return *m;
}

/// Segments by source module; never freed, since kernels keep pointers to them.
std::map<std::string, std::unique_ptr<segment>> &registry() {
  static auto *r = new std::map<std::string, std::unique_ptr<segment>>();
  return *r;
}

/// Placed ranges of the window: first address -> length (reserved bytes).
std::map<unsigned long long, unsigned long long> &placed_ranges() {
  static auto *p = new std::map<unsigned long long, unsigned long long>();
  return *p;
}

/// Segments that hold data and wait for placement.
std::vector<segment *> &pending() {
  static auto *p = new std::vector<segment *>();
  return *p;
}

/// The module named by the innermost module_scope; empty when none. Parsing holds
/// yasai_func::ptx_load_mutex, so one name serves all threads.
std::string g_scope_key;

/// Counters for YASAI_CVTA_REPORT.
std::atomic<unsigned long long> g_segments{0};
std::atomic<unsigned long long> g_bytes{0};
std::atomic<unsigned long long> g_out_of_range{0};

/// Appends the counters to the file YASAI_CVTA_REPORT names, at exit, when a segment was placed.
struct const_report {
  ~const_report() {
    const char *path = getenv("YASAI_CVTA_REPORT");
    if (path == nullptr || g_segments.load() == 0) return;
    char cwd[512] = "?";
    if (getcwd(cwd, sizeof cwd) == nullptr) cwd[0] = '?';
    FILE *f = fopen(path, "a");
    if (f == nullptr) return;
    fprintf(f, "const_segments=%llu const_bytes=%llu const_ld_past_end=%llu cwd=%s\n", g_segments.load(), g_bytes.load(),
            g_out_of_range.load(), cwd);
    fclose(f);
  }
} g_const_report;

/// Lowest free address of the window for a segment of len bytes, as a GPU's pool reuses space.
/// @param len Bytes, a multiple of kSegmentAlign.
/// @return The address, or 0 when the window has no room.
unsigned long long first_fit(unsigned long long len) {
  unsigned long long at = kWindowBase;
  for (const auto &r : placed_ranges()) {
    if (r.first >= at + len) break;
    at = r.first + r.second;
  }
  return at + len <= kWindowBase + kWindowSize ? at : 0;
}

/// Places one segment that holds data; the caller holds registry_mutex. A segment without data
/// keeps base 0, as on the GPU, and is placed when a constant arrives.
void place_locked(segment *seg) {
  if (seg->placed.load(std::memory_order_acquire) || seg->size == 0) return;
  static bool declared = false;
  if (!declared) {
    // The whole window is mapped: an access between or past the segments reads 0, as the GB10 did up
    // to 64 KiB past a segment (the farthest point measured).
    yasai_strict_addr::note_internal(kWindowBase, kWindowSize);
    declared = true;
  }
  const unsigned long long len = (seg->size + kSegmentAlign - 1) / kSegmentAlign * kSegmentAlign;
  const unsigned long long at = first_fit(len);
  if (at == 0) {
    fprintf(stderr,
            "yasai-const: the constant window [0x%llx, 0x%llx) has no room for the %llu bytes of %s; raise "
            "yasai_const::kWindowSize\n",
            kWindowBase, kWindowBase + kWindowSize, len, seg->key.c_str());
    abort();
  }
  placed_ranges()[at] = len;
  seg->reserved = len;
  seg->base.store(at, std::memory_order_relaxed);
  seg->placed.store(true, std::memory_order_release);
  g_segments.fetch_add(1, std::memory_order_relaxed);
  g_bytes.fetch_add(seg->size, std::memory_order_relaxed);
  printf("GPGPU-Sim PTX: constant segment of %s at 0x%llx (%llu bytes of data)\n", seg->key.c_str(), at, seg->size);
}

}  // namespace

module_scope::module_scope(const std::string &key) : m_previous(g_scope_key) { g_scope_key = key; }

module_scope::~module_scope() { g_scope_key = m_previous; }

segment *current(const char *parse_file) {
  const std::string key = !g_scope_key.empty() ? g_scope_key : std::string(parse_file ? parse_file : "");
  std::lock_guard<std::mutex> lk(registry_mutex());
  std::unique_ptr<segment> &slot = registry()[key];
  if (!slot) {
    slot.reset(new segment());
    slot->key = key;
  }
  return slot.get();
}

unsigned long long add_symbol(segment *seg, unsigned long long bytes, unsigned align, const char *name) {
  std::lock_guard<std::mutex> lk(registry_mutex());
  if (align == 0) align = 1;
  const unsigned long long offset = (seg->size + align - 1) / align * align;
  const unsigned long long end = offset + bytes;
  if (seg->placed.load(std::memory_order_acquire) && end > seg->reserved) {
    // A later piece of a module declared a constant that the first piece did not hold.
    fprintf(stderr,
            "yasai-const: %s grows the constant segment of %s to %llu bytes after it was placed with %llu\n", name,
            seg->key.c_str(), end, seg->reserved);
    abort();
  }
  if (seg->size == 0 && end > 0) pending().push_back(seg);
  seg->size = end;
  return offset;
}

void place_pending() {
  std::lock_guard<std::mutex> lk(registry_mutex());
  for (segment *seg : pending()) place_locked(seg);
  pending().clear();
}

unsigned long long base(segment *seg) {
  if (seg == nullptr || seg->size == 0) return 0;
  if (!seg->placed.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    place_locked(seg);
  }
  return seg->base.load(std::memory_order_relaxed);
}

unsigned long long host_address(const symbol *sym) {
  return base(sym->yasai_const_segment()) + sym->get_address();
}

segment *of_thread(ptx_thread_info *thread) {
  const function_info *entry = thread->get_kernel().entry();
  return entry ? entry->yasai_const_segment() : nullptr;
}

bool resolve_load(segment *seg, unsigned long long offset, unsigned long long bytes, unsigned long long *addr) {
  const unsigned long long b = base(seg);
  if (seg == nullptr || offset >= seg->size || bytes > seg->size - offset) {
    g_out_of_range.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  *addr = b + offset;
  return true;
}

bool contains(segment *seg, unsigned long long addr) {
  if (seg == nullptr || seg->size == 0) return false;
  const unsigned long long b = base(seg);
  return addr >= b && addr - b < seg->size;
}

void not_modeled(const char *what, const char *name, const std::string &detail) {
  fprintf(stderr, "yasai-const: not modeled: %s (%s; %s)\n", what, name, detail.c_str());
  printf("yasai-const: not modeled: %s (%s; %s)\n", what, name, detail.c_str());
  fflush(stdout);
  if (const char *path = getenv("YASAI_CVTA_REPORT")) {
    char cwd[512] = "?";
    if (getcwd(cwd, sizeof cwd) == nullptr) cwd[0] = '?';
    if (FILE *f = fopen(path, "a")) {
      fprintf(f, "const_not_modeled=%s name=%s detail=%s cwd=%s\n", what, name, detail.c_str(), cwd);
      fclose(f);
    }
  }
  abort();
}

}  // namespace yasai_const
