// shared_devmem.h: device memory as shareable host memory (yasai fakeenv simulation backend).
//
// Each rank is its own process with one simulated GPU. Communication libraries share device
// memory between ranks through CUDA IPC (cudaIpcGetMemHandle / cudaIpcOpenMemHandle), through the
// virtual memory management API (cuMemCreate, cuMemExportToShareableHandle with POSIX file
// descriptors, cuMemImportFromShareableHandle, cuMemMap) and through multicast objects
// (cuMulticastCreate, cuMulticastBindMem, multimem.* instructions). RDMA NICs (rxe) register
// device buffers with ibv_reg_mr. This module backs all of them with memfd segments:
//   * Device memory (cuMemCreate allocations once mapped, and cudaMalloc memory once exported by
//     legacy IPC) is a memfd mapped MAP_SHARED at its device address, which comes from the simulator's device heap. The device
//     address therefore equals the host virtual address (identity mapping, as for mapped host
//     memory), and the range is registered in host_window as device memory, so kernels, copies,
//     host threads and other processes see the same bytes with acquire/release visibility.
//     cudaMalloc memory starts as private anonymous memory at its device address, so reading a
//     page that was never written maps the zero page and costs no memory (a read fault on a
//     memfd hole allocates the page); YASAI_DEVMEM_EAGER_MEMFD=1 maps a memfd at once. Such an
//     allocation becomes a segment at the same address on its first legacy IPC export, on
//     cuPointerSetAttribute(SYNC_MEMOPS) (the GPUDirect RDMA idiom before a registration) and
//     through yasai_devmem_prepare_pin(), which the verbs recorder calls before ibv_reg_mr, so
//     that a pinning registration never holds pages that a later conversion replaces.
//     YASAI_DEVMEM_HOST=0 restores the simulator's own storage for cudaMalloc (legacy IPC then
//     moves an allocation into a memfd when it is first exported).
//   * A segment is reached from another process through its owner's descriptor, by the path
//     /proc/<owner pid>/fd/<fd> (the "ref"). Legacy IPC handles and multicast bindings carry the
//     ref; cuMem export hands out a duplicate descriptor that the importer receives over a Unix
//     socket. The peer maps the segment at a fresh device address of its own heap (it does not
//     reuse the owner's address). Refs work between processes of one PID namespace (one
//     container), which is where one node's ranks run.
//   * A multicast object is a memfd that holds a table of bound memory (ref and offsets per
//     binding). Its virtual addresses are registered in host_window as a multicast range;
//     multimem.* instructions and stores reach every bound copy, loads read the calling process's.
//   * memfd memory is freed when the last descriptor and mapping go away, so a crashed rank
//     leaves nothing behind in /dev/shm. Only touched pages take memory.
//   * After fork (zygote launch) a child moves the parent's allocations into its own segments,
//     so parent and child do not share device memory.
#ifndef SHARED_DEVMEM_H
#define SHARED_DEVMEM_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace shared_devmem {

struct phys;  // physical allocation or multicast object (opaque)

/// Reads `bytes` bytes of simulator device memory at `addr` into `out` (used when legacy IPC
/// moves a heap allocation into a segment, with YASAI_DEVMEM_HOST=0).
typedef std::function<void(uint64_t addr, size_t bytes, void *out)> read_fn;

/// Most devices a multicast object can have.
constexpr int kMaxMcDevices = 72;

/// Whether cudaMalloc returns memfd-backed host memory (YASAI_DEVMEM_HOST, default 1).
/// @return True unless YASAI_DEVMEM_HOST=0.
bool devmem_host_enabled();

/// Granularity of physical allocations, mappings and multicast objects (2 MiB, as on GPUs).
size_t granularity();

/// cudaMalloc in host memory mode: maps `size` bytes (rounded up to a page) of private anonymous
/// memory at device address `va` and records the allocation; its first legacy IPC export moves it
/// into a segment. With YASAI_DEVMEM_EAGER_MEMFD=1 a segment is mapped at once.
/// @param va Page-aligned device heap address that owns the rounded length.
/// @param size Length in bytes.
/// @return True on success; on failure nothing is mapped or recorded.
bool alloc_device(uint64_t va, size_t size);

/// Marks the calling thread as executing device work (the simulation thread for one turn of its
/// loop, a memset or a copy into device memory) while the object lives. A conversion of an
/// anonymous allocation into a segment (ipc_export, share_range) waits until no thread is inside
/// such a scope and keeps new ones out until it has copied the allocation and replaced its
/// mapping, so no store lands in the old pages after the copy. Scopes nest on one thread.
struct exec_scope {
  /// Enters the scope (waits while a conversion runs).
  exec_scope();
  /// Leaves the scope.
  ~exec_scope();
  exec_scope(const exec_scope &) = delete;
  exec_scope &operator=(const exec_scope &) = delete;
};

/// Creates a physical allocation of `size` bytes (rounded up to the granularity).
/// @param size Requested size in bytes.
/// @return The allocation, or nullptr when the segment cannot be created.
phys *create(size_t size);

/// Imports a physical allocation or multicast object from a POSIX file descriptor exported by
/// any process (the descriptor is duplicated; the caller keeps its own).
/// @param fd File descriptor of a segment.
/// @return The object, or nullptr when fd is not a usable segment.
phys *import_fd(int fd);

/// Exports a physical allocation or multicast object as a new POSIX file descriptor.
/// @param p Object to export.
/// @return A descriptor the caller owns, or -1.
int export_fd(phys *p);

/// Adds a reference to an object (one per handle the program holds).
/// @param p Object.
void retain(phys *p);

/// Drops a reference; the segment is closed at zero.
/// @param p Object.
void release(phys *p);

/// The live object a handle value names.
/// @param handle Handle value (a phys pointer).
/// @return The object, or nullptr.
phys *from_handle(uint64_t handle);

/// Size of an object in bytes.
/// @param p Object.
/// @return Size.
size_t size_of(phys *p);

/// Whether an object is a multicast object.
/// @param p Object.
/// @return True for multicast objects.
bool is_multicast(phys *p);

/// Reserves a device virtual address range whose addresses come from the caller (the simulator's
/// device heap): the host address range is claimed with an inaccessible mapping so that later
/// mappings can be placed there.
/// @param va First address (aligned to the granularity).
/// @param size Length in bytes (multiple of the granularity).
/// @return True on success.
bool reserve(uint64_t va, size_t size);

/// Releases a reserved range.
/// @param va First address passed to reserve().
/// @param size Its length.
void unreserve(uint64_t va, size_t size);

/// Maps `size` bytes of object `p`, starting at `offset`, at device address `va`.
/// @param va Device (and host) address inside a reserved range.
/// @param size Length in bytes.
/// @param offset Offset in the object.
/// @param p Physical allocation or multicast object.
/// @return True on success.
bool map(uint64_t va, size_t size, size_t offset, phys *p);

/// Unmaps a range mapped with map(); the range stays reserved.
/// @param va First address passed to map().
/// @param size Its length.
/// @return True when a mapping was removed.
bool unmap(uint64_t va, size_t size);

/// Finds the mapping that holds address `va`.
/// @param va Any address inside a mapping.
/// @param p Receives the mapped object.
/// @param base Receives the first address of the mapping.
/// @param size Receives the length of the mapping.
/// @param offset Receives the object offset of the mapping's first byte.
/// @return True when va lies in a mapping.
bool mapping_at(uint64_t va, phys **p, uint64_t *base, size_t *size, size_t *offset);

/// Records an allocation of the simulator's own storage (cudaMalloc with YASAI_DEVMEM_HOST=0),
/// for range queries and IPC.
/// @param base First address.
/// @param size Length in bytes.
void note_alloc(uint64_t base, size_t size);

/// Forgets an allocation and unmaps its segment, if any.
/// @param base First address.
/// @return True when an allocation started at base.
bool forget_alloc(uint64_t base);

/// Finds the allocation or mapping that holds `addr`.
/// @param addr Any device address.
/// @param base Receives the first address.
/// @param size Receives the length.
/// @return True when addr lies in a known allocation or mapping.
bool range_of(uint64_t addr, uint64_t *base, size_t *size);

/// Legacy IPC export of the allocation that holds `addr`: returns its segment ref (moving a
/// storage allocation into a segment first, once).
/// @param addr Address inside a cudaMalloc allocation.
/// @param read_old Reads a storage allocation's current contents.
/// @param ref Receives the segment ref (/proc/<pid>/fd/<fd>).
/// @param base Receives the allocation's first address.
/// @param size Receives the allocation's length.
/// @return True on success.
bool ipc_export(uint64_t addr, const read_fn &read_old, std::string *ref, uint64_t *base,
                size_t *size);

/// Whether a legacy IPC export of the allocation that holds `addr` copies it into a new segment
/// first (the allocation has none yet); the caller then synchronizes the device, so that no
/// kernel writes the allocation during the copy.
/// @param addr Address inside a cudaMalloc allocation.
/// @return True when the export will copy.
bool export_needs_copy(uint64_t addr);

/// Whether any cudaMalloc allocation overlapping [addr, addr + bytes) is still anonymous memory.
/// @param addr First address.
/// @param bytes Length (0 counts as 1).
/// @return True when share_range() would copy.
bool range_needs_copy(uint64_t addr, size_t bytes);

/// Moves every anonymous cudaMalloc allocation overlapping [addr, addr + bytes) into a segment
/// mapped at the same address, before an API that pins or shares the pages (an RDMA
/// registration) sees them. The caller has synchronized the device.
/// @param addr First address.
/// @param bytes Length (0 counts as 1).
/// @return True when every overlapping allocation is a segment afterwards.
bool share_range(uint64_t addr, size_t bytes);

/// Legacy IPC open: maps the segment `ref` of `size` bytes at device address `va` (taken by the
/// caller from the device heap).
/// @param ref Segment ref from ipc_export() in another process.
/// @param size Length in bytes.
/// @param va Device address to map at (page aligned).
/// @return True on success.
bool ipc_open(const char *ref, size_t size, uint64_t va);

/// Creates a small shared segment for an interprocess event's counters.
/// @param bytes Length.
/// @param ref Receives the segment ref.
/// @return Host address of the mapping, or nullptr.
void *small_shared(size_t bytes, std::string *ref);

/// Maps a segment by ref (another process's small shared segment).
/// @param ref Segment ref.
/// @param bytes Length to map.
/// @return Host address of the mapping, or nullptr.
void *open_shared(const char *ref, size_t bytes);

/// Creates a multicast object for `num_devices` devices of `size` bytes each.
/// @param num_devices Devices that will bind memory.
/// @param size Bytes per device.
/// @return The object, or nullptr.
phys *mc_create(unsigned num_devices, size_t size);

/// Binds `size` bytes of physical allocation `mem` at `mem_offset` to multicast offset
/// `mc_offset` for the calling process's device.
/// @param mc Multicast object.
/// @param mc_offset Offset in the multicast object.
/// @param mem Physical allocation.
/// @param mem_offset Offset in the allocation.
/// @param size Length in bytes.
/// @return True on success.
bool mc_bind(phys *mc, size_t mc_offset, phys *mem, size_t mem_offset, size_t size);

/// Removes the calling process's bindings that overlap [mc_offset, mc_offset + size).
/// @param mc Multicast object.
/// @param mc_offset Offset in the multicast object.
/// @param size Length in bytes.
void mc_unbind(phys *mc, size_t mc_offset, size_t size);

/// Host addresses (valid in this process) of every device's copy of multicast bytes
/// [off, off + bytes).
/// @param mc Multicast object (the mc field of a host_window multicast range).
/// @param off Offset in the multicast object.
/// @param bytes Length of the access.
/// @param addrs Receives one host address per bound device.
/// @param max Capacity of addrs.
/// @return Number of addresses written.
int mc_targets(void *mc, uint64_t off, size_t bytes, uint64_t *addrs, int max);

/// Host address of the calling process's own copy of multicast bytes [off, off + bytes), or of
/// the first bound copy when this process bound none.
/// @param mc Multicast object.
/// @param off Offset in the multicast object.
/// @param bytes Length of the access.
/// @return Host address, or 0 when nothing is bound there.
uint64_t mc_local_target(void *mc, uint64_t off, size_t bytes);

}  // namespace shared_devmem

#endif  // SHARED_DEVMEM_H
