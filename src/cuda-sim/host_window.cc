// host_window.cc: see host_window.h.
#include "host_window.h"
#include "fio_window.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace host_window {
namespace {

/// What the map keeps per registered range.
struct entry {
  size_t bytes;        ///< length of the range
  range_kind kind;     ///< what the range holds
  void *mc;            ///< multicast object of a multicast range
  uint64_t mc_offset;  ///< multicast object offset of the first byte
};

/// Registered ranges: first address to entry.
std::map<uint64_t, entry> &ranges() {
  static auto *m = new std::map<uint64_t, entry>;  // never destroyed: used during exit
  return *m;
}

/// Guards ranges(); lookups take it shared.
std::shared_mutex &ranges_mu() {
  static auto *m = new std::shared_mutex;
  return *m;
}

/// Number of registered ranges; lets contains() skip the lock when nothing is registered.
std::atomic<size_t> g_count{0};

/// Multicast hooks installed by shared_devmem.
mc_read_fn g_mc_read = nullptr;
/// See g_mc_read.
mc_write_fn g_mc_write = nullptr;

/// Whether an access of this size at this address can be a single atomic operation.
/// @param addr Address.
/// @param bytes Size.
/// @return True for naturally aligned 1, 2, 4 or 8 byte accesses.
bool atomic_size(uint64_t addr, size_t bytes) {
  return (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8) && (addr % bytes) == 0;
}

#if defined(__x86_64__)
/// 16-byte compare-and-swap (lock cmpxchg16b), atomic across processes.
/// @param p 16-byte aligned address.
/// @param e In: expected low and high words; out: the words found.
/// @param lo Desired low word.
/// @param hi Desired high word.
/// @return True when the swap happened.
bool cas16(void *p, uint64_t e[2], uint64_t lo, uint64_t hi) {
  bool ok;
  asm volatile("lock cmpxchg16b %1\n\tsete %0"
               : "=q"(ok), "+m"(*static_cast<volatile unsigned __int128 *>(p)), "+a"(e[0]),
                 "+d"(e[1])
               : "b"(lo), "c"(hi)
               : "memory", "cc");
  return ok;
}
#endif

/// Whether a 16-byte access is a single atomic operation here (aligned, x86-64). NCCL's LL and
/// LL128 protocols and vector loads/stores of packed data and flags need the whole vector to
/// become visible at once.
/// @param addr Address.
/// @param bytes Size.
/// @return True for aligned 16-byte accesses on x86-64.
bool atomic16(uint64_t addr, size_t bytes) {
#if defined(__x86_64__)
  return bytes == 16 && (addr % 16) == 0;
#else
  (void)addr;
  (void)bytes;
  return false;
#endif
}

#if defined(__x86_64__)
/// Whether an aligned 16-byte MOVDQA load is atomic on this processor. Intel SDM Vol. 3A
/// section 9.1.1 guarantees it for processors that enumerate AVX (CPUID.01H:ECX.AVX[bit 28]),
/// and the AMD APM Vol. 2 section 7.3.2 gives the same guarantee for AMD processors that report
/// AVX; GCC's libatomic relies on both (PR104688). Other vendors keep lock cmpxchg16b.
/// YASAI_HW_LOAD16_CAS=1 forces lock cmpxchg16b.
/// @return True on Intel or AMD processors with AVX.
bool load16_plain() {
  static const bool on = [] {
    if (const char *v = getenv("YASAI_HW_LOAD16_CAS"))
      if (v[0] == '1') return false;
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(0, &a, &b, &c, &d)) return false;
    char vendor[13];
    memcpy(vendor, &b, 4);
    memcpy(vendor + 4, &d, 4);
    memcpy(vendor + 8, &c, 4);
    vendor[12] = 0;
    const bool known = strcmp(vendor, "GenuineIntel") == 0 || strcmp(vendor, "AuthenticAMD") == 0;
    if (!known || !__get_cpuid(1, &a, &b, &c, &d)) return false;
    return (c & bit_AVX) != 0;
  }();
  return on;
}

/// Atomic aligned 16-byte load with MOVDQA (see load16_plain). Unlike lock cmpxchg16b it does
/// not write, so a load from a never-written page maps the zero page instead of committing a
/// private page.
/// @param p 16-byte aligned address.
/// @param out Destination of the 16 bytes.
void load16(const void *p, void *out) {
  typedef long long v2di __attribute__((vector_size(16)));
  v2di v;
  asm volatile("movdqa %1, %0" : "=x"(v) : "m"(*static_cast<const volatile v2di *>(p)) : "memory");
  memcpy(out, &v, 16);
}
#endif

/// Whether stores of zero over zero may be skipped (see try_write): YASAI_DEVMEM_ZERO_ELIDE is
/// not 0 and no warp trace is recorded (YASAI_TRACE_DIR unset or empty), so a trace run stores
/// every byte.
/// @return True unless YASAI_DEVMEM_ZERO_ELIDE=0 or YASAI_TRACE_DIR is set.
bool zero_elide() {
  static const bool on = [] {
    const char *v = getenv("YASAI_DEVMEM_ZERO_ELIDE");
    const char *t = getenv("YASAI_TRACE_DIR");
    return !(v && v[0] == '0') && !(t && t[0]);
  }();
  return on;
}

/// Number of stores skipped by skip_zero_store (see yasai_devmem_zero_skipped).
std::atomic<uint64_t> g_zero_skipped{0};

/// Depth of copy_zero_skip_scope objects on this thread.
thread_local int t_copy_scope = 0;

/// Whether every byte of a buffer is zero.
/// @param p Buffer.
/// @param bytes Length in bytes.
/// @return True when all `bytes` bytes are zero.
bool all_zero(const void *p, size_t bytes) {
  const unsigned char *c = static_cast<const unsigned char *>(p);
  size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    uint64_t w;
    memcpy(&w, c + i, 8);
    if (w) return false;
  }
  for (; i < bytes; i++)
    if (c[i]) return false;
  return true;
}

/// Page size for the chunked zero-store elision of long writes.
constexpr size_t kElideChunk = 4096;

/// Finds the range that holds [addr, addr + bytes); the caller holds ranges_mu().
/// @param addr First address.
/// @param bytes Length.
/// @return Iterator to the range, or ranges().end().
std::map<uint64_t, entry>::const_iterator find_locked(uint64_t addr, size_t bytes) {
  const auto &m = ranges();
  auto it = m.upper_bound(addr);
  if (it == m.begin()) return m.end();
  --it;
  if (addr >= it->first && addr + bytes <= it->first + it->second.bytes) return it;
  return m.end();
}

}  // namespace

void add(uint64_t base, size_t bytes, range_kind kind, void *mc, uint64_t mc_offset) {
  if (bytes == 0) return;
  std::unique_lock<std::shared_mutex> lock(ranges_mu());
  auto &m = ranges();
  // Drop ranges that overlap the new one.
  auto it = m.lower_bound(base);
  if (it != m.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second.bytes > base) it = prev;
  }
  while (it != m.end() && it->first < base + bytes) {
    // A replaced range usually means two registrations disagree about an address range.
    if (getenv("YASAI_HW_DEBUG"))
      fprintf(stderr, "host_window: [0x%llx, +%zu) kind %d replaces [0x%llx, +%zu) kind %d\n",
              (unsigned long long)base, bytes, int(kind), (unsigned long long)it->first,
              it->second.bytes, int(it->second.kind));
    it = m.erase(it);
  }
  if (getenv("YASAI_HW_DEBUG"))
    fprintf(stderr, "host_window: add [0x%llx, +%zu) kind %d\n", (unsigned long long)base, bytes,
            int(kind));
  m[base] = entry{bytes, kind, mc, mc_offset};
  g_count.store(m.size(), std::memory_order_release);
}

bool remove(uint64_t base) {
  std::unique_lock<std::shared_mutex> lock(ranges_mu());
  bool removed = ranges().erase(base) > 0;
  g_count.store(ranges().size(), std::memory_order_release);
  return removed;
}

bool contains(uint64_t addr, size_t bytes) {
  if (g_count.load(std::memory_order_acquire) == 0) return false;
  std::shared_lock<std::shared_mutex> lock(ranges_mu());
  return find_locked(addr, bytes) != ranges().end();
}

bool lookup(uint64_t addr, size_t bytes, range_info *out) {
  if (g_count.load(std::memory_order_acquire) == 0) return false;
  std::shared_lock<std::shared_mutex> lock(ranges_mu());
  auto it = find_locked(addr, bytes);
  if (it == ranges().end()) return false;
  if (out) *out = range_info{it->first, it->second.bytes, it->second.kind, it->second.mc,
                             it->second.mc_offset};
  return true;
}

size_t size_at(uint64_t base) {
  std::shared_lock<std::shared_mutex> lock(ranges_mu());
  auto it = ranges().find(base);
  return it == ranges().end() ? 0 : it->second.bytes;
}

void set_multicast_hooks(mc_read_fn rd, mc_write_fn wr) {
  g_mc_read = rd;
  g_mc_write = wr;
}

void read(uint64_t addr, size_t bytes, void *out) {
  void *p = reinterpret_cast<void *>(addr);
  if (atomic_size(addr, bytes)) {
    switch (bytes) {
      case 1: { uint8_t v = __atomic_load_n(static_cast<uint8_t *>(p), __ATOMIC_ACQUIRE); memcpy(out, &v, 1); return; }
      case 2: { uint16_t v = __atomic_load_n(static_cast<uint16_t *>(p), __ATOMIC_ACQUIRE); memcpy(out, &v, 2); return; }
      case 4: { uint32_t v = __atomic_load_n(static_cast<uint32_t *>(p), __ATOMIC_ACQUIRE); memcpy(out, &v, 4); return; }
      default: { uint64_t v = __atomic_load_n(static_cast<uint64_t *>(p), __ATOMIC_ACQUIRE); memcpy(out, &v, 8); return; }
    }
  }
#if defined(__x86_64__)
  if (atomic16(addr, bytes)) {
    if (load16_plain()) {
      load16(p, out);
      return;
    }
    uint64_t e[2] = {0, 0};
    cas16(p, e, 0, 0);  // stores 0 over 0 (no change) or returns the current value
    memcpy(out, e, 16);
    return;
  }
#endif
  std::atomic_thread_fence(std::memory_order_acquire);
  memcpy(out, p, bytes);
}

/// Performs a weak store or a copy into device memory, skipping the parts where both the stored
/// bytes and the memory are zero (the memory then already holds the value). The caller has
/// checked that skipping is allowed (see try_write).
/// @param addr Device (identity-mapped host) address.
/// @param bytes Length in bytes.
/// @param in Source buffer.
static void skip_zero_store(uint64_t addr, size_t bytes, const void *in) {
  if (bytes <= 16) {
    unsigned char cur[16];
    if (all_zero(in, bytes)) {
      read(addr, bytes, cur);  // never writes (MOVDQA for 16 bytes), so no page is committed
      if (all_zero(cur, bytes)) {
        g_zero_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    write(addr, bytes, in);
    return;
  }
  // Long writes (copies, not single atomic operations): page by page.
  const unsigned char *src = static_cast<const unsigned char *>(in);
  unsigned char *dst = reinterpret_cast<unsigned char *>(addr);
  std::atomic_thread_fence(std::memory_order_release);
  size_t off = 0;
  while (off < bytes) {
    size_t n = kElideChunk - ((addr + off) % kElideChunk);
    if (n > bytes - off) n = bytes - off;
    if (all_zero(src + off, n) && all_zero(dst + off, n))
      g_zero_skipped.fetch_add(1, std::memory_order_relaxed);
    else
      memcpy(dst + off, src + off, n);
    off += n;
  }
  std::atomic_thread_fence(std::memory_order_release);
}

copy_zero_skip_scope::copy_zero_skip_scope() { ++t_copy_scope; }

copy_zero_skip_scope::~copy_zero_skip_scope() { --t_copy_scope; }

bool in_copy_zero_skip_scope() { return t_copy_scope > 0; }

uint64_t zero_skipped_count() { return g_zero_skipped.load(std::memory_order_relaxed); }

void write(uint64_t addr, size_t bytes, const void *in) {
  void *p = reinterpret_cast<void *>(addr);
  if (atomic_size(addr, bytes)) {
    switch (bytes) {
      case 1: { uint8_t v; memcpy(&v, in, 1); __atomic_store_n(static_cast<uint8_t *>(p), v, __ATOMIC_RELEASE); return; }
      case 2: { uint16_t v; memcpy(&v, in, 2); __atomic_store_n(static_cast<uint16_t *>(p), v, __ATOMIC_RELEASE); return; }
      case 4: { uint32_t v; memcpy(&v, in, 4); __atomic_store_n(static_cast<uint32_t *>(p), v, __ATOMIC_RELEASE); return; }
      default: { uint64_t v; memcpy(&v, in, 8); __atomic_store_n(static_cast<uint64_t *>(p), v, __ATOMIC_RELEASE); return; }
    }
  }
#if defined(__x86_64__)
  if (atomic16(addr, bytes)) {
    uint64_t d[2];
    memcpy(d, in, 16);
    uint64_t e[2] = {0, 0};
    while (!cas16(p, e, d[0], d[1])) {
    }
    return;
  }
#endif
  std::atomic_thread_fence(std::memory_order_release);
  memcpy(p, in, bytes);
  std::atomic_thread_fence(std::memory_order_release);
}

bool try_read(uint64_t addr, size_t bytes, void *out) {
  range_info r;
  if (!lookup(addr, bytes, &r)) return false;
  if (r.kind == range_kind::multicast) {
    if (g_mc_read) g_mc_read(r.mc, r.mc_offset + (addr - r.base), bytes, out);
    else memset(out, 0, bytes);
    return true;
  }
  // yasai: an unordered FIO window drains its due writes into shared memory before the read, then
  // overlays this process's own pending writes so the issuing kernel reads them in program order.
  const bool fio = fio_window::any();
  if (fio) fio_window::before_load(addr, bytes);
  read(addr, bytes, out);
  if (fio) fio_window::overlay(addr, bytes, out);
  return true;
}

bool try_write(uint64_t addr, size_t bytes, const void *in, bool weak) {
  range_info r;
  if (!lookup(addr, bytes, &r)) return false;
  if (r.kind == range_kind::multicast) {
    if (g_mc_write) g_mc_write(r.mc, r.mc_offset + (addr - r.base), bytes, in);
    return true;
  }
  // yasai: a store into an unordered FIO window is buffered (made visible later, out of order),
  // not written now. It goes before the zero-store skip, so the window model and the FIO transport
  // record see every store, zeros over zeros included.
  if (fio_window::any() && fio_window::on_store(addr, bytes, in)) return true;
  // yasai: the zero-store skip comes last, after every hook that must see the store (an
  // unordered FIO window's buffering goes before it), and only for device memory.
  if (weak && r.kind == range_kind::device && zero_elide()) {
    skip_zero_store(addr, bytes, in);
    return true;
  }
  write(addr, bytes, in);
  return true;
}

bool compare_exchange(uint64_t addr, size_t bytes, uint64_t *expected, uint64_t desired) {
  void *p = reinterpret_cast<void *>(addr);
  switch (bytes) {
    case 2: {
      uint16_t e = uint16_t(*expected);
      bool ok = __atomic_compare_exchange_n(static_cast<uint16_t *>(p), &e, uint16_t(desired),
                                            false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
      *expected = e;
      return ok;
    }
    case 4: {
      uint32_t e = uint32_t(*expected);
      bool ok = __atomic_compare_exchange_n(static_cast<uint32_t *>(p), &e, uint32_t(desired),
                                            false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
      *expected = e;
      return ok;
    }
    default: {
      uint64_t e = *expected;
      bool ok = __atomic_compare_exchange_n(static_cast<uint64_t *>(p), &e, desired, false,
                                            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
      *expected = e;
      return ok;
    }
  }
}

}  // namespace host_window
