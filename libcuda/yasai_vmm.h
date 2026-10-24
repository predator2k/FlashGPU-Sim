// yasai_vmm.h: CUDA runtime and driver entry points for device memory shared between rank
// processes (legacy IPC, IPC events, the cuMem* virtual memory management API, multicast objects
// and pointer queries), on top of src/cuda-sim/shared_devmem. The driver functions that the
// simulator did not define before (cuMemCreate, cuMemMap, cuMulticastCreate, ...) are defined
// in yasai_vmm.cc with C linkage, so cudaGetDriverEntryPoint hands them out; the functions that
// had stubs in cuda_runtime_api.cc forward to the helpers declared here.
#ifndef YASAI_VMM_H
#define YASAI_VMM_H

#include <cstddef>
#include <cstdint>

class gpgpu_t;

namespace yasai_vmm {

/// Allocates device heap addresses aligned to `align` (a power of two); the length is rounded up
/// to the alignment so that the allocation owns whole pages.
/// @param gpu Simulated GPU.
/// @param size Length in bytes.
/// @param align Alignment in bytes.
/// @return First address.
uint64_t dev_alloc(gpgpu_t *gpu, size_t size, size_t align);

/// cudaMalloc: allocates page-aligned device memory and records it for range queries and IPC.
/// @param gpu Simulated GPU.
/// @param size Length in bytes.
/// @return First address (0 on failure).
uint64_t malloc_device(gpgpu_t *gpu, size_t size);

/// cudaFree of a cudaMalloc allocation (drops its shared segment, if any).
/// @param ptr First address of the allocation.
void free_device(uint64_t ptr);

/// cudaIpcGetMemHandle: fills the 64-byte handle for the allocation that holds `ptr`.
/// @param gpu Simulated GPU (its storage holds the current contents).
/// @param handle 64-byte handle.
/// @param ptr Address inside a cudaMalloc allocation.
/// @return True on success.
bool ipc_get_mem_handle(gpgpu_t *gpu, void *handle, uint64_t ptr);

/// cudaIpcOpenMemHandle: maps another process's allocation.
/// @param gpu Simulated GPU (supplies the device address).
/// @param handle 64-byte handle from ipc_get_mem_handle.
/// @param ptr Receives the device address.
/// @return True on success.
bool ipc_open_mem_handle(gpgpu_t *gpu, const void *handle, uint64_t *ptr);

/// cudaIpcCloseMemHandle.
/// @param ptr Address returned by ipc_open_mem_handle.
/// @return True when ptr was an opened handle.
bool ipc_close_mem_handle(uint64_t ptr);

/// Creates the shared counters of an interprocess event.
/// @param counters Receives the two counters (issued, updates).
/// @param name Receives the segment name (64 bytes).
/// @return True on success.
bool ipc_event_create(unsigned int **counters, char *name);

/// Maps the shared counters of another process's interprocess event.
/// @param handle 64-byte event handle.
/// @param counters Receives the two counters.
/// @param name Receives the segment name (64 bytes).
/// @return True on success.
bool ipc_event_open(const void *handle, unsigned int **counters, char *name);

/// Fills a 64-byte event handle from an interprocess event's segment name.
/// @param handle 64-byte handle.
/// @param name Segment name.
void ipc_event_handle(void *handle, const char *name);

/// cuMemGetAddressRange.
/// @param addr Any device address.
/// @param base Receives the allocation's first address.
/// @param size Receives its length.
/// @return True when addr lies in a known allocation or mapping.
bool address_range(uint64_t addr, uint64_t *base, size_t *size);

/// cuPointerGetAttribute.
/// @param data Receives the attribute value.
/// @param attr CUpointer_attribute value.
/// @param ptr Pointer.
/// @param ctx Current context (for CU_POINTER_ATTRIBUTE_CONTEXT).
/// @return A CUresult value.
int pointer_attribute(void *data, int attr, uint64_t ptr, void *ctx);

/// Whether a pointer is pinned host memory (host_window kind host) rather than device memory.
/// @param ptr Pointer.
/// @return True for pinned host memory.
bool is_pinned_host(uint64_t ptr);

}  // namespace yasai_vmm

#endif  // YASAI_VMM_H
