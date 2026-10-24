// shared_devmem.cc: see shared_devmem.h.
#include "shared_devmem.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "host_window.h"

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000  // Linux 4.17 and later
#endif

namespace shared_devmem {

namespace {

/// Granularity of cuMem allocations, mappings and multicast objects (2 MiB).
constexpr size_t kGranularity = size_t(2) << 20;
/// Host page size; cudaMalloc segments are rounded to it.
constexpr size_t kPage = 4096;
/// Chunk size of copies between segments (all-zero chunks are skipped).
constexpr size_t kCopyChunk = size_t(64) << 10;
/// Bindings a multicast object can hold (several per device are allowed).
constexpr int kMaxMcSlots = 4 * kMaxMcDevices;
/// Length of a segment ref, including the terminating zero.
constexpr int kRefLen = 48;
/// Magic of a phys object (guards from_handle).
constexpr uint64_t kPhysMagic = 0x5941534149504859ull;  // "YASAIPHY"
/// Magic of a multicast table.
constexpr uint64_t kMcMagic = 0x59415341494d4332ull;  // "YASAIMC2"

/// One binding of a multicast object, in the shared table.
struct mc_slot {
  uint32_t state;        ///< 1 when the binding is valid (written last, read with acquire)
  uint32_t pad;          ///< alignment
  uint64_t owner;        ///< token of the process that bound the memory
  uint64_t mc_offset;    ///< first multicast offset of the binding
  uint64_t mem_offset;   ///< offset in the bound segment
  uint64_t size;         ///< length of the binding
  char ref[kRefLen];     ///< ref of the bound segment
};

/// Multicast table, at the start of a multicast object's segment.
struct mc_header {
  uint64_t magic;         ///< kMcMagic
  uint64_t size;          ///< bytes per device
  uint32_t num_devices;   ///< devices of the team
  uint32_t nslots;        ///< slots handed out so far (atomic)
  mc_slot slots[kMaxMcSlots];  ///< bindings
};

}  // namespace

/// Physical allocation or multicast object.
struct phys {
  uint64_t magic;          ///< kPhysMagic while alive
  int fd;                  ///< this process's descriptor of the segment
  size_t size;             ///< length of the segment (bytes per device for multicast objects)
  bool multicast;          ///< whether this is a multicast object
  mc_header *hdr;          ///< mapped table of a multicast object
  std::atomic<int> refs;   ///< handles, mappings and bindings that keep the object alive
};

namespace {

/// One device virtual address mapping (cuMemMap or an opened legacy IPC handle).
struct mapping {
  size_t size;    ///< length
  size_t offset;  ///< object offset of the first byte
  phys *p;        ///< mapped object
};

/// One cudaMalloc allocation.
struct allocation {
  size_t size;   ///< length
  phys *seg;     ///< its segment (host memory mode or after a legacy IPC export), or nullptr
};

/// Peer segment mapped by this process to reach another device's copy of multicast memory.
struct peer_map {
  char *base;   ///< host address of the mapping
  size_t size;  ///< length
};

/// All module state, guarded by mu.
struct state {
  std::mutex mu;                               ///< guards everything below
  std::set<phys *> live;                       ///< live objects
  std::map<uint64_t, mapping> maps;            ///< mappings by first address
  std::map<uint64_t, allocation> allocs;       ///< cudaMalloc allocations by first address
  std::map<std::string, peer_map> peers;       ///< peer segments by ref
  uint64_t token = 0;                          ///< random token of this process
};

/// The module state (never destroyed: used during exit).
state &st() {
  static state *s = new state;
  return *s;
}

/// Rounds n up to a multiple of a (a power of two).
/// @param n Value.
/// @param a Alignment.
/// @return Rounded value.
size_t round_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

/// Ref of a descriptor of this process.
/// @param fd Descriptor.
/// @return /proc/<pid>/fd/<fd>.
std::string ref_of(int fd) {
  char b[kRefLen];
  snprintf(b, sizeof(b), "/proc/%d/fd/%d", int(getpid()), fd);
  return b;
}

/// Draws this process's token from /dev/urandom.
void new_token() {
  state &s = st();
  s.token = 0;
  FILE *f = fopen("/dev/urandom", "rb");
  if (f) {
    if (fread(&s.token, sizeof(s.token), 1, f) != 1) s.token = 0;
    fclose(f);
  }
  if (s.token == 0) s.token = uint64_t(getpid()) * 0x9e3779b97f4a7c15ull + 1;
}

/// Copies the non-zero chunks of [src, src + n) to dst (dst is fresh, zero-filled memory).
/// @param dst Destination.
/// @param src Source.
/// @param n Length.
void copy_nonzero(char *dst, const char *src, size_t n) {
  static const char zero[kCopyChunk] = {0};
  for (size_t off = 0; off < n; off += kCopyChunk) {
    const size_t c = std::min(kCopyChunk, n - off);
    if (memcmp(src + off, zero, c) != 0) memcpy(dst + off, src + off, c);
  }
}

/// Creates a sized memfd.
/// @param size Length.
/// @return Descriptor, or -1.
int new_memfd(size_t size) {
  int fd = memfd_create("yasai_devmem", MFD_CLOEXEC);
  if (fd < 0) {
    fprintf(stderr, "shared_devmem: memfd_create failed: %s\n", strerror(errno));
    return -1;
  }
  if (ftruncate(fd, off_t(size)) != 0) {
    fprintf(stderr, "shared_devmem: ftruncate to %zu failed: %s\n", size, strerror(errno));
    close(fd);
    return -1;
  }
  return fd;
}

/// Wraps a descriptor in a new object with one reference; the caller holds st().mu.
/// @param fd Descriptor (owned by the object).
/// @param size Length.
/// @return The object.
phys *wrap_locked(int fd, size_t size) {
  phys *p = new phys;
  p->magic = kPhysMagic;
  p->fd = fd;
  p->size = size;
  p->multicast = false;
  p->hdr = nullptr;
  p->refs.store(1);
  st().live.insert(p);
  return p;
}

/// Drops one reference; the caller holds st().mu.
/// @param p Object.
void release_locked(phys *p) {
  if (p->refs.fetch_sub(1) != 1) return;
  st().live.erase(p);
  if (p->hdr) munmap(p->hdr, sizeof(mc_header));
  close(p->fd);
  p->magic = 0;
  delete p;
}

/// Places object bytes at a fixed address and registers the range; the caller holds st().mu.
/// @param va Address.
/// @param size Length.
/// @param offset Object offset.
/// @param fd Segment descriptor.
/// @param noreplace Whether the address must not be mapped yet (no reservation).
/// @return True on success.
bool place_locked(uint64_t va, size_t size, size_t offset, int fd, bool noreplace) {
  int flags = MAP_SHARED | (noreplace ? MAP_FIXED_NOREPLACE : MAP_FIXED);
  void *r = mmap(reinterpret_cast<void *>(va), size, PROT_READ | PROT_WRITE, flags, fd,
                 off_t(offset));
  if (r == MAP_FAILED || reinterpret_cast<uint64_t>(r) != va) {
    fprintf(stderr, "shared_devmem: mmap at 0x%llx (%zu bytes) failed: %s\n",
            (unsigned long long)va, size, r == MAP_FAILED ? strerror(errno) : "moved");
    if (r != MAP_FAILED) munmap(r, size);
    return false;
  }
  host_window::add(va, size, host_window::range_kind::device);
  return true;
}

/// Opens another process's segment by ref and maps all of it; the caller holds st().mu.
/// @param ref Segment ref.
/// @return The mapping, or nullptr.
peer_map *peer_locked(const char *ref) {
  state &s = st();
  auto it = s.peers.find(ref);
  if (it != s.peers.end()) return &it->second;
  int fd = open(ref, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    fprintf(stderr, "shared_devmem: cannot open bound segment %s: %s\n", ref, strerror(errno));
    return nullptr;
  }
  struct stat sb;
  if (fstat(fd, &sb) != 0) {
    close(fd);
    return nullptr;
  }
  void *b = mmap(nullptr, size_t(sb.st_size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (b == MAP_FAILED) return nullptr;
  peer_map pm{static_cast<char *>(b), size_t(sb.st_size)};
  return &s.peers.emplace(ref, pm).first->second;
}

/// Multicast read hook for host_window: reads the calling process's copy.
/// @param mc Multicast object.
/// @param off Offset in the object.
/// @param bytes Length.
/// @param out Destination.
void hook_read(void *mc, uint64_t off, size_t bytes, void *out) {
  uint64_t a = mc_local_target(mc, off, bytes);
  if (a) host_window::read(a, bytes, out);
  else memset(out, 0, bytes);
}

/// Multicast write hook for host_window: a store to a multicast address reaches every copy.
/// @param mc Multicast object.
/// @param off Offset in the object.
/// @param bytes Length.
/// @param in Source.
void hook_write(void *mc, uint64_t off, size_t bytes, const void *in) {
  uint64_t addrs[kMaxMcDevices];
  int n = mc_targets(mc, off, bytes, addrs, kMaxMcDevices);
  for (int i = 0; i < n; i++) host_window::write(addrs[i], bytes, in);
}

/// In a forked child: moves every cudaMalloc segment inherited from the parent into a segment
/// of the child's own (the parent keeps using the old ones), and draws a new token. Refs held by
/// peers name the parent's descriptors and stay valid for the parent.
void after_fork_child() {
  state &s = st();
  new (&s.mu) std::mutex();  // the parent may have held it
  new_token();
  for (auto &kv : s.allocs) {
    allocation &a = kv.second;
    if (!a.seg) continue;
    const size_t len = round_up(a.size, kPage);
    int fd = new_memfd(len);
    if (fd < 0) continue;
    void *tmp = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (tmp == MAP_FAILED) {
      close(fd);
      continue;
    }
    copy_nonzero(static_cast<char *>(tmp), reinterpret_cast<const char *>(kv.first), len);
    munmap(tmp, len);
    mmap(reinterpret_cast<void *>(kv.first), len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
         fd, 0);
    close(a.seg->fd);
    a.seg->fd = fd;
  }
  if (!s.maps.empty())
    fprintf(stderr, "shared_devmem: %zu cuMem mappings made before fork stay shared with the "
            "parent\n", s.maps.size());
}

/// One-time setup: token, descriptor limit, fork handler, multicast hooks.
void init_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    new_token();
    // Every allocation keeps a descriptor open: raise the soft limit to the hard one.
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < rl.rlim_max) {
      rl.rlim_cur = rl.rlim_max;
      setrlimit(RLIMIT_NOFILE, &rl);
    }
    pthread_atfork(nullptr, nullptr, after_fork_child);
    host_window::set_multicast_hooks(hook_read, hook_write);
  });
}

}  // namespace

bool devmem_host_enabled() {
  static const bool on = [] {
    const char *v = getenv("YASAI_DEVMEM_HOST");
    return !(v && atoi(v) == 0);
  }();
  return on;
}

size_t granularity() { return kGranularity; }

bool alloc_device(uint64_t va, size_t size) {
  init_once();
  const size_t len = round_up(size ? size : 1, kPage);
  int fd = new_memfd(len);
  if (fd < 0) return false;
  std::lock_guard<std::mutex> lock(st().mu);
  if (!place_locked(va, len, 0, fd, true)) {
    close(fd);
    return false;
  }
  st().allocs[va] = allocation{size ? size : 1, wrap_locked(fd, len)};
  return true;
}

phys *create(size_t size) {
  init_once();
  const size_t len = round_up(size ? size : 1, kGranularity);
  int fd = new_memfd(len);
  if (fd < 0) return nullptr;
  std::lock_guard<std::mutex> lock(st().mu);
  return wrap_locked(fd, len);
}

phys *import_fd(int fd) {
  init_once();
  int d = fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (d < 0) return nullptr;
  struct stat sb;
  if (fstat(d, &sb) != 0) {
    close(d);
    return nullptr;
  }
  // A multicast object starts with its table.
  mc_header *h = nullptr;
  if (size_t(sb.st_size) >= sizeof(mc_header)) {
    void *m = mmap(nullptr, sizeof(mc_header), PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
    if (m != MAP_FAILED) {
      if (__atomic_load_n(&static_cast<mc_header *>(m)->magic, __ATOMIC_ACQUIRE) == kMcMagic)
        h = static_cast<mc_header *>(m);
      else
        munmap(m, sizeof(mc_header));
    }
  }
  std::lock_guard<std::mutex> lock(st().mu);
  phys *p = wrap_locked(d, size_t(sb.st_size));
  if (h) {
    p->multicast = true;
    p->hdr = h;
    p->size = size_t(h->size);
  }
  return p;
}

int export_fd(phys *p) { return p ? fcntl(p->fd, F_DUPFD_CLOEXEC, 0) : -1; }

void retain(phys *p) {
  if (p) p->refs.fetch_add(1);
}

void release(phys *p) {
  if (!p) return;
  std::lock_guard<std::mutex> lock(st().mu);
  release_locked(p);
}

phys *from_handle(uint64_t handle) {
  phys *p = reinterpret_cast<phys *>(handle);
  std::lock_guard<std::mutex> lock(st().mu);
  return st().live.count(p) ? p : nullptr;
}

size_t size_of(phys *p) { return p ? p->size : 0; }

bool is_multicast(phys *p) { return p && p->multicast; }

bool reserve(uint64_t va, size_t size) {
  void *r = mmap(reinterpret_cast<void *>(va), size, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
  if (r == MAP_FAILED || reinterpret_cast<uint64_t>(r) != va) {
    fprintf(stderr, "shared_devmem: cannot reserve 0x%llx (%zu bytes): %s\n",
            (unsigned long long)va, size, r == MAP_FAILED ? strerror(errno) : "moved");
    if (r != MAP_FAILED) munmap(r, size);
    return false;
  }
  return true;
}

void unreserve(uint64_t va, size_t size) { munmap(reinterpret_cast<void *>(va), size); }

bool map(uint64_t va, size_t size, size_t offset, phys *p) {
  if (!p || size == 0) return false;
  std::lock_guard<std::mutex> lock(st().mu);
  if (p->multicast) {
    host_window::add(va, size, host_window::range_kind::multicast, p, offset);
  } else if (!place_locked(va, size, offset, p->fd, false)) {
    return false;
  }
  p->refs.fetch_add(1);
  st().maps[va] = mapping{size, offset, p};
  return true;
}

bool unmap(uint64_t va, size_t size) {
  (void)size;
  std::lock_guard<std::mutex> lock(st().mu);
  state &s = st();
  auto it = s.maps.find(va);
  if (it == s.maps.end()) return false;
  phys *p = it->second.p;
  const size_t len = it->second.size;
  host_window::remove(va);
  if (!p->multicast) {
    // Back to an inaccessible reservation.
    mmap(reinterpret_cast<void *>(va), len, PROT_NONE,
         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
  }
  s.maps.erase(it);
  release_locked(p);
  return true;
}

bool mapping_at(uint64_t va, phys **p, uint64_t *base, size_t *size, size_t *offset) {
  std::lock_guard<std::mutex> lock(st().mu);
  auto &m = st().maps;
  auto it = m.upper_bound(va);
  if (it == m.begin()) return false;
  --it;
  if (va >= it->first + it->second.size) return false;
  if (p) *p = it->second.p;
  if (base) *base = it->first;
  if (size) *size = it->second.size;
  if (offset) *offset = it->second.offset;
  return true;
}

void note_alloc(uint64_t base, size_t size) {
  std::lock_guard<std::mutex> lock(st().mu);
  st().allocs[base] = allocation{size, nullptr};
}

bool forget_alloc(uint64_t base) {
  std::lock_guard<std::mutex> lock(st().mu);
  state &s = st();
  auto it = s.allocs.find(base);
  if (it == s.allocs.end()) return false;
  if (phys *p = it->second.seg) {
    const size_t len = round_up(it->second.size, kPage);
    host_window::remove(base);
    munmap(reinterpret_cast<void *>(base), len);
    release_locked(p);
  }
  s.allocs.erase(it);
  return true;
}

bool range_of(uint64_t addr, uint64_t *base, size_t *size) {
  std::lock_guard<std::mutex> lock(st().mu);
  state &s = st();
  auto ia = s.allocs.upper_bound(addr);
  if (ia != s.allocs.begin()) {
    --ia;
    if (addr < ia->first + ia->second.size) {
      *base = ia->first;
      *size = ia->second.size;
      return true;
    }
  }
  auto im = s.maps.upper_bound(addr);
  if (im != s.maps.begin()) {
    --im;
    if (addr < im->first + im->second.size) {
      *base = im->first;
      *size = im->second.size;
      return true;
    }
  }
  return false;
}

bool ipc_export(uint64_t addr, const read_fn &read_old, std::string *ref, uint64_t *base,
                size_t *size) {
  init_once();
  std::lock_guard<std::mutex> lock(st().mu);
  state &s = st();
  auto it = s.allocs.upper_bound(addr);
  if (it == s.allocs.begin()) return false;
  --it;
  if (addr >= it->first + it->second.size) return false;
  allocation &a = it->second;
  const uint64_t b = it->first;
  if (!a.seg) {
    // Storage allocation (YASAI_DEVMEM_HOST=0): move its contents into a segment mapped at the
    // same address, so the owner's pointers stay valid.
    if (b % kPage) {
      fprintf(stderr, "shared_devmem: allocation 0x%llx is not page aligned\n",
              (unsigned long long)b);
      return false;
    }
    const size_t len = round_up(a.size, kPage);
    int fd = new_memfd(len);
    if (fd < 0) return false;
    void *tmp = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (tmp == MAP_FAILED) {
      close(fd);
      return false;
    }
    std::vector<char> buf(kCopyChunk);
    for (size_t off = 0; off < a.size; off += kCopyChunk) {
      const size_t n = std::min(kCopyChunk, a.size - off);
      read_old(b + off, n, buf.data());
      copy_nonzero(static_cast<char *>(tmp) + off, buf.data(), n);
    }
    munmap(tmp, len);
    if (!place_locked(b, len, 0, fd, true)) {
      close(fd);
      return false;
    }
    a.seg = wrap_locked(fd, len);
  }
  *ref = ref_of(a.seg->fd);
  *base = b;
  *size = a.size;
  return true;
}

bool ipc_open(const char *ref, size_t size, uint64_t va) {
  init_once();
  int fd = open(ref, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    fprintf(stderr, "shared_devmem: cannot open IPC segment %s: %s\n", ref, strerror(errno));
    return false;
  }
  std::lock_guard<std::mutex> lock(st().mu);
  const size_t len = round_up(size, kPage);
  if (!place_locked(va, len, 0, fd, true)) {
    close(fd);
    return false;
  }
  st().maps[va] = mapping{len, 0, wrap_locked(fd, len)};  // the mapping holds the reference
  return true;
}

void *small_shared(size_t bytes, std::string *ref) {
  init_once();
  int fd = new_memfd(round_up(bytes, kPage));
  if (fd < 0) return nullptr;
  void *m = mmap(nullptr, round_up(bytes, kPage), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) {
    close(fd);
    return nullptr;
  }
  *ref = ref_of(fd);  // the descriptor stays open for the life of the process
  return m;
}

void *open_shared(const char *ref, size_t bytes) {
  int fd = open(ref, O_RDWR | O_CLOEXEC);
  if (fd < 0) return nullptr;
  void *m = mmap(nullptr, round_up(bytes, kPage), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return m == MAP_FAILED ? nullptr : m;
}

phys *mc_create(unsigned num_devices, size_t size) {
  if (num_devices == 0 || int(num_devices) > kMaxMcDevices) return nullptr;
  init_once();
  const size_t len = round_up(sizeof(mc_header), kPage);
  int fd = new_memfd(len);
  if (fd < 0) return nullptr;
  void *h = mmap(nullptr, sizeof(mc_header), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (h == MAP_FAILED) {
    close(fd);
    return nullptr;
  }
  mc_header *hdr = static_cast<mc_header *>(h);
  hdr->size = round_up(size, kGranularity);
  hdr->num_devices = num_devices;
  __atomic_store_n(&hdr->magic, kMcMagic, __ATOMIC_RELEASE);
  std::lock_guard<std::mutex> lock(st().mu);
  phys *p = wrap_locked(fd, size_t(hdr->size));
  p->multicast = true;
  p->hdr = hdr;
  return p;
}

bool mc_bind(phys *mc, size_t mc_offset, phys *mem, size_t mem_offset, size_t size) {
  if (!mc || !mc->hdr || !mem || mem->multicast) return false;
  if (mc_offset + size > mc->hdr->size || mem_offset + size > mem->size) return false;
  init_once();
  uint32_t i = __atomic_fetch_add(&mc->hdr->nslots, 1u, __ATOMIC_ACQ_REL);
  if (int(i) >= kMaxMcSlots) return false;
  mc_slot &sl = mc->hdr->slots[i];
  sl.owner = st().token;
  sl.mc_offset = mc_offset;
  sl.mem_offset = mem_offset;
  sl.size = size;
  snprintf(sl.ref, sizeof(sl.ref), "%s", ref_of(mem->fd).c_str());
  retain(mem);  // the binding keeps the descriptor (and so the ref) valid
  __atomic_store_n(&sl.state, 1u, __ATOMIC_RELEASE);
  return true;
}

void mc_unbind(phys *mc, size_t mc_offset, size_t size) {
  if (!mc || !mc->hdr) return;
  init_once();
  const uint64_t me = st().token;
  uint32_t n = __atomic_load_n(&mc->hdr->nslots, __ATOMIC_ACQUIRE);
  for (uint32_t i = 0; i < n && int(i) < kMaxMcSlots; i++) {
    mc_slot &sl = mc->hdr->slots[i];
    if (__atomic_load_n(&sl.state, __ATOMIC_ACQUIRE) != 1 || sl.owner != me) continue;
    if (sl.mc_offset >= mc_offset + size || sl.mc_offset + sl.size <= mc_offset) continue;
    __atomic_store_n(&sl.state, 0u, __ATOMIC_RELEASE);
  }
}

int mc_targets(void *mcv, uint64_t off, size_t bytes, uint64_t *addrs, int max) {
  phys *mc = static_cast<phys *>(mcv);
  if (!mc || !mc->hdr) return 0;
  std::lock_guard<std::mutex> lock(st().mu);
  int n = 0;
  uint32_t ns = __atomic_load_n(&mc->hdr->nslots, __ATOMIC_ACQUIRE);
  for (uint32_t i = 0; i < ns && int(i) < kMaxMcSlots && n < max; i++) {
    mc_slot &sl = mc->hdr->slots[i];
    if (__atomic_load_n(&sl.state, __ATOMIC_ACQUIRE) != 1) continue;
    if (off < sl.mc_offset || off + bytes > sl.mc_offset + sl.size) continue;
    peer_map *pm = peer_locked(sl.ref);
    if (!pm) continue;
    addrs[n++] = reinterpret_cast<uint64_t>(pm->base) + sl.mem_offset + (off - sl.mc_offset);
  }
  return n;
}

uint64_t mc_local_target(void *mcv, uint64_t off, size_t bytes) {
  phys *mc = static_cast<phys *>(mcv);
  if (!mc || !mc->hdr) return 0;
  std::lock_guard<std::mutex> lock(st().mu);
  const uint64_t me = st().token;
  uint64_t first = 0;
  uint32_t ns = __atomic_load_n(&mc->hdr->nslots, __ATOMIC_ACQUIRE);
  for (uint32_t i = 0; i < ns && int(i) < kMaxMcSlots; i++) {
    mc_slot &sl = mc->hdr->slots[i];
    if (__atomic_load_n(&sl.state, __ATOMIC_ACQUIRE) != 1) continue;
    if (off < sl.mc_offset || off + bytes > sl.mc_offset + sl.size) continue;
    peer_map *pm = peer_locked(sl.ref);
    if (!pm) continue;
    uint64_t a = reinterpret_cast<uint64_t>(pm->base) + sl.mem_offset + (off - sl.mc_offset);
    if (sl.owner == me) return a;
    if (!first) first = a;
  }
  return first;
}

}  // namespace shared_devmem
