// host_window.cc: see host_window.h.
#include "host_window.h"

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>

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
  while (it != m.end() && it->first < base + bytes) it = m.erase(it);
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
    uint64_t e[2] = {0, 0};
    cas16(p, e, 0, 0);  // stores 0 over 0 (no change) or returns the current value
    memcpy(out, e, 16);
    return;
  }
#endif
  std::atomic_thread_fence(std::memory_order_acquire);
  memcpy(out, p, bytes);
}

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
  read(addr, bytes, out);
  return true;
}

bool try_write(uint64_t addr, size_t bytes, const void *in) {
  range_info r;
  if (!lookup(addr, bytes, &r)) return false;
  if (r.kind == range_kind::multicast) {
    if (g_mc_write) g_mc_write(r.mc, r.mc_offset + (addr - r.base), bytes, in);
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
