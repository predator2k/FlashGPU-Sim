// host_window.cc: see host_window.h.
#include "host_window.h"

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>

namespace host_window {
namespace {

/// Registered ranges: first address to length.
std::map<uint64_t, size_t> &ranges() {
  static auto *m = new std::map<uint64_t, size_t>;  // never destroyed: used during exit
  return *m;
}

/// Guards ranges(); lookups take it shared.
std::shared_mutex &ranges_mu() {
  static auto *m = new std::shared_mutex;
  return *m;
}

/// Number of registered ranges; lets contains() skip the lock when nothing is registered.
std::atomic<size_t> g_count{0};

/// Whether an access of this size at this address can be a single atomic operation.
/// @param addr Address.
/// @param bytes Size.
/// @return True for naturally aligned 1, 2, 4 or 8 byte accesses.
bool atomic_size(uint64_t addr, size_t bytes) {
  return (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8) && (addr % bytes) == 0;
}

}  // namespace

void add(uint64_t base, size_t bytes) {
  if (bytes == 0) return;
  std::unique_lock<std::shared_mutex> lock(ranges_mu());
  auto &m = ranges();
  // Drop ranges that overlap the new one.
  auto it = m.lower_bound(base);
  if (it != m.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second > base) it = prev;
  }
  while (it != m.end() && it->first < base + bytes) it = m.erase(it);
  m[base] = bytes;
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
  const auto &m = ranges();
  auto it = m.upper_bound(addr);
  if (it == m.begin()) return false;
  --it;
  return addr >= it->first && addr + bytes <= it->first + it->second;
}

size_t size_at(uint64_t base) {
  std::shared_lock<std::shared_mutex> lock(ranges_mu());
  auto it = ranges().find(base);
  return it == ranges().end() ? 0 : it->second;
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
  std::atomic_thread_fence(std::memory_order_release);
  memcpy(p, in, bytes);
  std::atomic_thread_fence(std::memory_order_release);
}

}  // namespace host_window
