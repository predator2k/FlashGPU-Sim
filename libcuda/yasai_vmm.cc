// yasai_vmm.cc: see yasai_vmm.h.
#include "yasai_vmm.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/host_window.h"
#include "../src/cuda-sim/memory.h"
#include "../src/cuda-sim/shared_devmem.h"

class CUctx_st;
struct gpgpu_context;
gpgpu_context *GPGPU_Context();
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
/// Simulated GPU of the current context (defined in cuda_runtime_api.cc).
gpgpu_t *yasai_current_gpu();

namespace {

/// CUresult values used here (CUDA 13 cuda.h).
enum : int {
  kSuccess = 0,
  kInvalidValue = 1,
  kOutOfMemory = 2,
  kNotSupported = 801,
  kNotFound = 500,
  kInvalidHandle = 400,
};

/// CUmemAllocationHandleType bits (CUDA 13 cuda.h).
enum : int { kHandlePosixFd = 0x1, kHandleFabric = 0x8 };

/// Magic at the start of a yasai IPC memory or event handle.
constexpr uint32_t kIpcMagic = 0x4d485359u;  // "YSHM"
/// Bytes of a CUDA IPC handle.
constexpr size_t kIpcHandleBytes = 64;
/// Offset of the segment name in an IPC handle (after the magic and the size).
constexpr size_t kIpcNameOffset = 12;
/// Bytes of an interprocess event's shared counters (issued, updates).
constexpr size_t kEventBytes = 8;

/// CUmemLocation (CUDA 13 cuda.h).
struct yasai_CUmemLocation {
  int type;  ///< CUmemLocationType
  int id;    ///< device ordinal or NUMA node
};

/// CUmemAllocationProp (CUDA 13 cuda.h).
struct yasai_CUmemAllocationProp {
  int type;                          ///< CUmemAllocationType
  int requestedHandleTypes;          ///< CUmemAllocationHandleType bits
  yasai_CUmemLocation location;      ///< where the memory lives
  void *win32HandleMetaData;         ///< Windows only
  struct {
    unsigned char compressionType;       ///< compression hint
    unsigned char gpuDirectRDMACapable;  ///< RDMA flag
    unsigned short usage;                ///< usage bits
    unsigned char reserved[4];           ///< reserved
  } allocFlags;                      ///< allocation flags
};

/// CUmemAccessDesc (CUDA 13 cuda.h).
struct yasai_CUmemAccessDesc {
  yasai_CUmemLocation location;  ///< device the access applies to
  int flags;                     ///< CUmemAccess_flags
};

/// CUmulticastObjectProp (CUDA 13 cuda.h).
struct yasai_CUmulticastObjectProp {
  unsigned int numDevices;     ///< devices of the team
  size_t size;                 ///< bytes per device
  unsigned long long handleTypes;  ///< exportable handle types
  unsigned long long flags;    ///< must be zero
};

/// Serializes legacy IPC opens (device address allocation plus mapping).
std::mutex g_ipc_mu;

/// Rounds n up to a multiple of a (a power of two).
/// @param n Value.
/// @param a Alignment.
/// @return Rounded value.
size_t round_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

/// Object of a handle value.
/// @param h CUmemGenericAllocationHandle value.
/// @return The object, or nullptr.
shared_devmem::phys *obj(unsigned long long h) { return shared_devmem::from_handle(h); }

}  // namespace

namespace yasai_vmm {

uint64_t dev_alloc(gpgpu_t *gpu, size_t size, size_t align) {
  const uint64_t top = gpu->dev_malloc_top();
  const uint64_t aligned = (top + align - 1) & ~uint64_t(align - 1);
  if (aligned > top) gpu->gpu_malloc(size_t(aligned - top));
  return reinterpret_cast<uint64_t>(gpu->gpu_malloc(round_up(size ? size : 1, align)));
}

uint64_t malloc_device(gpgpu_t *gpu, size_t size) {
  const size_t page = 4096;
  uint64_t p = dev_alloc(gpu, size, page);
  if (!p) return 0;
  // Host memory mode: a memfd at the device address. If it cannot be placed there (address in
  // use), the allocation stays in the simulator's storage.
  if (shared_devmem::devmem_host_enabled() && shared_devmem::alloc_device(p, size)) return p;
  shared_devmem::note_alloc(p, size ? size : 1);
  return p;
}

void free_device(uint64_t ptr) { shared_devmem::forget_alloc(ptr); }

bool ipc_get_mem_handle(gpgpu_t *gpu, void *handle, uint64_t ptr) {
  std::string name;
  uint64_t base = 0;
  size_t size = 0;
  memory_space *mem = gpu->get_global_memory();
  auto rd = [mem](uint64_t a, size_t n, void *out) { mem->read(a, n, out); };
  if (!shared_devmem::ipc_export(ptr, rd, &name, &base, &size)) return false;
  unsigned char *h = static_cast<unsigned char *>(handle);
  memset(h, 0, kIpcHandleBytes);
  memcpy(h, &kIpcMagic, 4);
  const uint64_t sz = size;
  memcpy(h + 4, &sz, 8);
  snprintf(reinterpret_cast<char *>(h + kIpcNameOffset), kIpcHandleBytes - kIpcNameOffset, "%s",
           name.c_str());
  return true;
}

bool ipc_open_mem_handle(gpgpu_t *gpu, const void *handle, uint64_t *ptr) {
  const unsigned char *h = static_cast<const unsigned char *>(handle);
  uint32_t magic;
  memcpy(&magic, h, 4);
  if (magic != kIpcMagic) return false;
  uint64_t size;
  memcpy(&size, h + 4, 8);
  char name[kIpcHandleBytes];
  memcpy(name, h + kIpcNameOffset, kIpcHandleBytes - kIpcNameOffset);
  name[kIpcHandleBytes - kIpcNameOffset - 1] = 0;
  std::lock_guard<std::mutex> lock(g_ipc_mu);
  const size_t gran = shared_devmem::granularity();
  const uint64_t va = dev_alloc(gpu, round_up(size, gran), gran);
  if (!shared_devmem::ipc_open(name, size, va)) return false;
  *ptr = va;
  return true;
}

bool ipc_close_mem_handle(uint64_t ptr) {
  shared_devmem::phys *p = nullptr;
  uint64_t base = 0;
  size_t size = 0, off = 0;
  if (!shared_devmem::mapping_at(ptr, &p, &base, &size, &off) || base != ptr) return false;
  shared_devmem::unmap(ptr, size);
  shared_devmem::unreserve(ptr, size);
  return true;
}

bool ipc_event_create(unsigned int **counters, char *name) {
  std::string ref;
  void *m = shared_devmem::small_shared(kEventBytes, &ref);
  if (!m) return false;
  snprintf(name, kIpcHandleBytes - kIpcNameOffset, "%s", ref.c_str());
  *counters = static_cast<unsigned int *>(m);
  return true;
}

bool ipc_event_open(const void *handle, unsigned int **counters, char *name) {
  const unsigned char *h = static_cast<const unsigned char *>(handle);
  uint32_t magic;
  memcpy(&magic, h, 4);
  if (magic != kIpcMagic) return false;
  memcpy(name, h + kIpcNameOffset, kIpcHandleBytes - kIpcNameOffset);
  name[kIpcHandleBytes - kIpcNameOffset - 1] = 0;
  void *m = shared_devmem::open_shared(name, kEventBytes);
  if (!m) return false;
  *counters = static_cast<unsigned int *>(m);
  return true;
}

void ipc_event_handle(void *handle, const char *name) {
  unsigned char *h = static_cast<unsigned char *>(handle);
  memset(h, 0, kIpcHandleBytes);
  memcpy(h, &kIpcMagic, 4);
  snprintf(reinterpret_cast<char *>(h + kIpcNameOffset), kIpcHandleBytes - kIpcNameOffset, "%s",
           name);
}

bool address_range(uint64_t addr, uint64_t *base, size_t *size) {
  if (shared_devmem::range_of(addr, base, size)) return true;
  host_window::range_info r;
  if (host_window::lookup(addr, 1, &r)) {
    *base = r.base;
    *size = r.bytes;
    return true;
  }
  return false;
}

bool is_pinned_host(uint64_t ptr) {
  host_window::range_info r;
  return host_window::lookup(ptr, 1, &r) && r.kind == host_window::range_kind::host;
}

int pointer_attribute(void *data, int attr, uint64_t ptr, void *ctx) {
  // CUpointer_attribute values (CUDA 13 cuda.h).
  enum {
    kContext = 1, kMemoryType = 2, kDevicePointer = 3, kHostPointer = 4, kSyncMemops = 6,
    kBufferId = 7, kIsManaged = 8, kDeviceOrdinal = 9, kLegacyIpc = 10, kRangeStart = 11,
    kRangeSize = 12, kMapped = 13, kAllowedHandles = 14, kGdrCapable = 15, kAccessFlags = 16,
    kMempool = 17, kMappingSize = 18, kMappingBase = 19, kBlockId = 20, kHwDecompress = 21,
  };
  // CUmemorytype values.
  enum { kMemHost = 1, kMemDevice = 2 };
  const bool host = is_pinned_host(ptr);
  uint64_t base = 0;
  size_t size = 0;
  const bool known = address_range(ptr, &base, &size);
  const bool device = !host && (known || ptr >= GLOBAL_HEAP_START) && ptr < 0x500000000000ull;
  if (!host && !device) return kInvalidValue;
  switch (attr) {
    case kContext: *static_cast<void **>(data) = ctx; return kSuccess;
    case kMemoryType: *static_cast<unsigned *>(data) = host ? kMemHost : kMemDevice; return kSuccess;
    case kDevicePointer: *static_cast<uint64_t *>(data) = ptr; return kSuccess;
    case kHostPointer:
      if (!host) return kInvalidValue;
      *static_cast<uint64_t *>(data) = ptr;
      return kSuccess;
    case kSyncMemops: *static_cast<int *>(data) = 1; return kSuccess;
    case kBufferId: *static_cast<unsigned long long *>(data) = known ? base : ptr; return kSuccess;
    case kIsManaged: *static_cast<int *>(data) = 0; return kSuccess;
    case kDeviceOrdinal: *static_cast<int *>(data) = 0; return kSuccess;
    case kLegacyIpc: *static_cast<int *>(data) = (device && known) ? 1 : 0; return kSuccess;
    case kRangeStart:
      if (!known) return kInvalidValue;
      *static_cast<uint64_t *>(data) = base;
      return kSuccess;
    case kRangeSize:
      if (!known) return kInvalidValue;
      *static_cast<size_t *>(data) = size;
      return kSuccess;
    case kMapped: *static_cast<int *>(data) = 1; return kSuccess;
    case kAllowedHandles: *static_cast<unsigned long long *>(data) = kHandlePosixFd; return kSuccess;
    case kGdrCapable: *static_cast<int *>(data) = 0; return kSuccess;
    case kAccessFlags: *static_cast<int *>(data) = 3; return kSuccess;  // PROT_READWRITE
    case kMempool: *static_cast<void **>(data) = nullptr; return kSuccess;
    case kMappingSize:
      if (!known) return kInvalidValue;
      *static_cast<size_t *>(data) = size;
      return kSuccess;
    case kMappingBase:
      if (!known) return kInvalidValue;
      *static_cast<uint64_t *>(data) = base;
      return kSuccess;
    case kBlockId: *static_cast<unsigned long long *>(data) = known ? base : ptr; return kSuccess;
    case kHwDecompress: *static_cast<int *>(data) = 0; return kSuccess;
    default: return kInvalidValue;
  }
}

}  // namespace yasai_vmm

// ---- driver API: virtual memory management and multicast (C linkage, found by
// cudaGetDriverEntryPoint) ----

extern "C" {

/// cuMemGetAllocationGranularity: 2 MiB for every request.
int cuMemGetAllocationGranularity(size_t *granularity, const yasai_CUmemAllocationProp *prop,
                                  int option) {
  (void)prop;
  (void)option;
  if (!granularity) return kInvalidValue;
  *granularity = shared_devmem::granularity();
  return kSuccess;
}

/// cuMemCreate: a physical allocation backed by a shared memory segment (device or host
/// location alike; both are identity mapped once mapped). Fabric handles are not supported.
int cuMemCreate(unsigned long long *handle, size_t size, const yasai_CUmemAllocationProp *prop,
                unsigned long long flags) {
  (void)flags;
  if (!handle || !prop) return kInvalidValue;
  if (prop->requestedHandleTypes & kHandleFabric) return kNotSupported;
  shared_devmem::phys *p = shared_devmem::create(size);
  if (!p) return kOutOfMemory;
  *handle = reinterpret_cast<unsigned long long>(p);
  return kSuccess;
}

/// cuMemRelease: drops the program's handle (mappings keep the memory alive).
int cuMemRelease(unsigned long long handle) {
  shared_devmem::phys *p = obj(handle);
  if (!p) return kInvalidHandle;
  shared_devmem::release(p);
  return kSuccess;
}

/// cuMemAddressReserve: device heap addresses claimed as an inaccessible host mapping.
int cuMemAddressReserve(unsigned long long *ptr, size_t size, size_t alignment,
                        unsigned long long addr, unsigned long long flags) {
  (void)addr;
  (void)flags;
  if (!ptr || size == 0) return kInvalidValue;
  const size_t gran = shared_devmem::granularity();
  size_t align = alignment > gran ? alignment : gran;
  const size_t len = round_up(size, gran);
  uint64_t va = yasai_vmm::dev_alloc(yasai_current_gpu(), len, align);
  if (!shared_devmem::reserve(va, len)) return kOutOfMemory;
  *ptr = va;
  return kSuccess;
}

/// cuMemAddressFree.
int cuMemAddressFree(unsigned long long ptr, size_t size) {
  shared_devmem::unreserve(ptr, round_up(size, shared_devmem::granularity()));
  return kSuccess;
}

/// cuMemMap: places the allocation (or multicast object) at the reserved addresses.
int cuMemMap(unsigned long long ptr, size_t size, size_t offset, unsigned long long handle,
             unsigned long long flags) {
  (void)flags;
  shared_devmem::phys *p = obj(handle);
  if (!p) return kInvalidHandle;
  return shared_devmem::map(ptr, size, offset, p) ? kSuccess : kInvalidValue;
}

/// cuMemUnmap.
int cuMemUnmap(unsigned long long ptr, size_t size) {
  return shared_devmem::unmap(ptr, size) ? kSuccess : kInvalidValue;
}

/// cuMemSetAccess: every mapping is readable and writable by the one device of the process.
int cuMemSetAccess(unsigned long long ptr, size_t size, const yasai_CUmemAccessDesc *desc,
                   size_t count) {
  (void)ptr;
  (void)size;
  (void)desc;
  (void)count;
  return kSuccess;
}

/// cuMemGetAccess: read-write.
int cuMemGetAccess(unsigned long long *flags, const yasai_CUmemLocation *location,
                   unsigned long long ptr) {
  (void)location;
  (void)ptr;
  if (flags) *flags = 3;
  return kSuccess;
}

/// cuMemExportToShareableHandle: POSIX file descriptors only.
int cuMemExportToShareableHandle(void *shareable, unsigned long long handle, int type,
                                 unsigned long long flags) {
  (void)flags;
  shared_devmem::phys *p = obj(handle);
  if (!p || !shareable) return kInvalidValue;
  if (type != kHandlePosixFd) return kNotSupported;
  int fd = shared_devmem::export_fd(p);
  if (fd < 0) return kInvalidValue;
  *static_cast<int *>(shareable) = fd;
  return kSuccess;
}

/// cuMemImportFromShareableHandle: a POSIX file descriptor from any rank process (the
/// descriptor value is passed in the pointer argument, as with the real driver).
int cuMemImportFromShareableHandle(unsigned long long *handle, void *os_handle, int type) {
  if (!handle) return kInvalidValue;
  if (type != kHandlePosixFd) return kNotSupported;
  int fd = int(reinterpret_cast<intptr_t>(os_handle));
  shared_devmem::phys *p = shared_devmem::import_fd(fd);
  if (!p) return kInvalidValue;
  *handle = reinterpret_cast<unsigned long long>(p);
  return kSuccess;
}

/// cuMemRetainAllocationHandle: the object mapped at an address (one more reference).
int cuMemRetainAllocationHandle(unsigned long long *handle, void *addr) {
  shared_devmem::phys *p = nullptr;
  if (!handle ||
      !shared_devmem::mapping_at(reinterpret_cast<uint64_t>(addr), &p, nullptr, nullptr, nullptr))
    return kInvalidValue;
  shared_devmem::retain(p);
  *handle = reinterpret_cast<unsigned long long>(p);
  return kSuccess;
}

/// cuMemGetAllocationPropertiesFromHandle: pinned device memory of device 0, POSIX fd export.
int cuMemGetAllocationPropertiesFromHandle(yasai_CUmemAllocationProp *prop,
                                           unsigned long long handle) {
  if (!prop || !obj(handle)) return kInvalidValue;
  memset(prop, 0, sizeof(*prop));
  prop->type = 1;           // CU_MEM_ALLOCATION_TYPE_PINNED
  prop->requestedHandleTypes = kHandlePosixFd;
  prop->location.type = 1;  // CU_MEM_LOCATION_TYPE_DEVICE
  prop->location.id = 0;
  return kSuccess;
}

/// cuMemGetHandleForAddressRange: DMA-BUF export is not supported (no GPUDirect RDMA).
int cuMemGetHandleForAddressRange(void *handle, unsigned long long ptr, size_t size, int type,
                                  unsigned long long flags) {
  (void)handle;
  (void)ptr;
  (void)size;
  (void)type;
  (void)flags;
  return kNotSupported;
}

/// cuMulticastGetGranularity: 2 MiB.
int cuMulticastGetGranularity(size_t *granularity, const yasai_CUmulticastObjectProp *prop,
                              int option) {
  (void)prop;
  (void)option;
  if (!granularity) return kInvalidValue;
  *granularity = shared_devmem::granularity();
  return kSuccess;
}

/// cuMulticastCreate.
int cuMulticastCreate(unsigned long long *handle, const yasai_CUmulticastObjectProp *prop) {
  if (!handle || !prop) return kInvalidValue;
  if (prop->handleTypes & kHandleFabric) return kNotSupported;
  shared_devmem::phys *p = shared_devmem::mc_create(prop->numDevices, prop->size);
  if (!p) return kOutOfMemory;
  *handle = reinterpret_cast<unsigned long long>(p);
  return kSuccess;
}

/// cuMulticastAddDevice: membership is implied by binding (each rank process binds its own
/// device's memory).
int cuMulticastAddDevice(unsigned long long mc, int dev) {
  (void)dev;
  return shared_devmem::is_multicast(obj(mc)) ? kSuccess : kInvalidHandle;
}

/// cuMulticastBindMem.
int cuMulticastBindMem(unsigned long long mc, size_t mc_offset, unsigned long long mem,
                       size_t mem_offset, size_t size, unsigned long long flags) {
  (void)flags;
  shared_devmem::phys *m = obj(mc), *p = obj(mem);
  if (!m || !p) return kInvalidHandle;
  return shared_devmem::mc_bind(m, mc_offset, p, mem_offset, size) ? kSuccess : kInvalidValue;
}

/// cuMulticastBindAddr: binds the memory mapped at a device address.
int cuMulticastBindAddr(unsigned long long mc, size_t mc_offset, unsigned long long memptr,
                        size_t size, unsigned long long flags) {
  (void)flags;
  shared_devmem::phys *m = obj(mc), *p = nullptr;
  uint64_t base = 0;
  size_t len = 0, off = 0;
  if (!m || !shared_devmem::mapping_at(memptr, &p, &base, &len, &off)) return kInvalidValue;
  return shared_devmem::mc_bind(m, mc_offset, p, off + (memptr - base), size) ? kSuccess
                                                                              : kInvalidValue;
}

/// cuMulticastUnbind.
int cuMulticastUnbind(unsigned long long mc, int dev, size_t mc_offset, size_t size) {
  (void)dev;
  shared_devmem::phys *m = obj(mc);
  if (!m) return kInvalidHandle;
  shared_devmem::mc_unbind(m, mc_offset, size);
  return kSuccess;
}

}  // extern "C"
