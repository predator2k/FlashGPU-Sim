
#include "kernel_info.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <map>
#include <mutex>
#include <vector>
#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/yasai_strict_addr.h"
#include "gpgpu-sim/gpu-sim.h"
#include "stream_manager.h"

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

kernel_info_t::kernel_info_t(dim3 gridDim, dim3 blockDim,
                             class function_info *entry,
                             unsigned long long streamID) {
  m_kernel_entry = entry;
  // yasai: every kernel has its dynamic shared memory base before it runs (no dynamic bytes
  // included; see function_info::yasai_place_dyn_smem).
  m_kernel_entry->yasai_place_dyn_smem();
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  m_streamID = streamID;
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;

  cache_config_set = false;
  m_dynamic_smem = 0;
}

/*A snapshot of the texture mappings needs to be stored in the kernel's info as
kernels should use the texture bindings seen at the time of launch and textures
 can be bound/unbound asynchronously with respect to streams. */
kernel_info_t::kernel_info_t(
    dim3 gridDim, dim3 blockDim, class function_info *entry,
    std::map<std::string, const struct cudaArray *> nameToCudaArray,
    std::map<std::string, const struct textureInfo *> nameToTextureInfo) {
  m_kernel_entry = entry;
  // yasai: every kernel has its dynamic shared memory base before it runs (no dynamic bytes
  // included; see function_info::yasai_place_dyn_smem).
  m_kernel_entry->yasai_place_dyn_smem();
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;

  cache_config_set = false;
  m_NameToCudaArray = nameToCudaArray;
  m_NameToTextureInfo = nameToTextureInfo;
  m_dynamic_smem = 0;
}

namespace {

/// yasai: guards the free lists of parameter copies and the creation of a copy.
/// @return The mutex.
std::mutex &param_window_mutex() {
  static std::mutex m;
  return m;
}

/// yasai: first address of the parameters' copies. They are global addresses (whichspace) outside
/// the device heap, which grows from GLOBAL_HEAP_START, and below the range where Linux places host
/// mappings, so that a program's allocations and memory counters do not see them.
constexpr unsigned long long kParamWindowStart = 0x600000000000ull;
/// yasai: alignment of a copy in bytes, enough for every .param alignment in the census.
constexpr unsigned long long kParamWindowAlign = 256;
/// yasai: next address for a new copy, guarded by param_window_mutex.
unsigned long long g_param_window_next = kParamWindowStart;

/// yasai: copies made, including reused ones.
std::atomic<unsigned long long> g_param_window_copies{0};
/// yasai: tensor maps read from a copy.
std::atomic<unsigned long long> g_param_window_tensor_maps{0};
/// yasai: prefetch instructions executed.
std::atomic<unsigned long long> g_prefetches{0};
/// yasai: end of the addresses given to copies so far; written under param_window_mutex.
std::atomic<unsigned long long> g_param_window_end{kParamWindowStart};

/// yasai: appends the counters and the working directory to the file YASAI_CVTA_REPORT names, at exit, when a counter is not zero.
struct param_window_report {
  ~param_window_report() {
    const char *path = getenv("YASAI_CVTA_REPORT");
    if (path == nullptr || g_param_window_copies.load() + g_prefetches.load() == 0) return;
    char cwd[512] = "?";
    if (getcwd(cwd, sizeof cwd) == nullptr) cwd[0] = '?';
    FILE *f = fopen(path, "a");
    if (f == nullptr) return;
    fprintf(f, "copies=%llu tensor_map_reads=%llu prefetches=%llu cwd=%s\n", g_param_window_copies.load(), g_param_window_tensor_maps.load(), g_prefetches.load(), cwd);
    fclose(f);
  }
} g_param_window_report;

/// yasai: copies of finished launches, by length, for the next launches: a long run has millions
/// of launches, and storage is never returned.
/// @return Length to addresses.
std::map<unsigned, std::vector<unsigned long long>> &param_window_free() {
  static auto *lists = new std::map<unsigned, std::vector<unsigned long long>>();
  return *lists;
}

}  // namespace

unsigned long long kernel_info_t::yasai_param_window(gpgpu_t *gpu) {
  unsigned long long base = m_yasai_param_window.load(std::memory_order_acquire);
  if (base != 0) return base;
  std::lock_guard<std::mutex> lk(param_window_mutex());
  base = m_yasai_param_window.load(std::memory_order_relaxed);
  if (base != 0) return base;
  const unsigned size = m_kernel_entry->yasai_param_block_size();
  if (size == 0) {
    fprintf(stderr, "yasai-cvta: kernel %s has no parameter block for cvta.param\n", name().c_str());
    abort();
  }
  std::vector<unsigned long long> &free_list = param_window_free()[size];
  if (free_list.empty()) {
    base = g_param_window_next;
    g_param_window_next += (size + kParamWindowAlign - 1) / kParamWindowAlign * kParamWindowAlign;
    g_param_window_end.store(g_param_window_next, std::memory_order_relaxed);
    yasai_strict_addr::note_internal(base, size);
  } else {
    base = free_list.back();
    free_list.pop_back();
  }
  // Word by word, since a parameter may cross a block of the parameter space (function_info::finalize).
  for (unsigned i = 0; i < size; i += 4) {
    const unsigned n = std::min(4u, size - i);
    unsigned word = 0;
    m_param_mem->read(i, n, &word);
    gpu->get_global_memory()->write(base + i, n, &word, NULL, NULL);
  }
  m_yasai_param_window_size = size;
  m_yasai_param_window.store(base, std::memory_order_release);
  g_param_window_copies.fetch_add(1, std::memory_order_relaxed);
  return base;
}

void yasai_cvta_report::note_tensor_map_read(unsigned long long addr) {
  if (addr >= kParamWindowStart && addr < g_param_window_end.load(std::memory_order_relaxed))
    g_param_window_tensor_maps.fetch_add(1, std::memory_order_relaxed);
}

void yasai_cvta_report::note_prefetch() { g_prefetches.fetch_add(1, std::memory_order_relaxed); }

bool kernel_info_t::yasai_in_param_window(unsigned long long addr) const {
  const unsigned long long base = m_yasai_param_window.load(std::memory_order_acquire);
  return base != 0 && addr >= base && addr - base <= m_yasai_param_window_size;
}

kernel_info_t::~kernel_info_t() {
  assert(m_active_threads.empty());
  destroy_cta_streams();
  delete m_param_mem;
  const unsigned long long window = m_yasai_param_window.load(std::memory_order_acquire);
  if (window != 0) {
    std::lock_guard<std::mutex> lk(param_window_mutex());
    param_window_free()[m_yasai_param_window_size].push_back(window);
  }
}

std::string kernel_info_t::name() const { return m_kernel_entry->get_name(); }

// Jin: parent and child kernel management for CDP
void kernel_info_t::set_parent(kernel_info_t *parent, dim3 parent_ctaid,
                               dim3 parent_tid) {
  m_parent_kernel = parent;
  m_parent_ctaid = parent_ctaid;
  m_parent_tid = parent_tid;
  parent->set_child(this);
}

void kernel_info_t::set_child(kernel_info_t *child) {
  m_child_kernels.push_back(child);
}

void kernel_info_t::remove_child(kernel_info_t *child) {
  assert(std::find(m_child_kernels.begin(), m_child_kernels.end(), child) !=
         m_child_kernels.end());
  m_child_kernels.remove(child);
}

bool kernel_info_t::is_finished() {
  if (done() && children_all_finished())
    return true;
  else
    return false;
}

bool kernel_info_t::children_all_finished() {
  if (!m_child_kernels.empty()) return false;

  return true;
}

void kernel_info_t::notify_parent_finished() {
  if (m_parent_kernel) {
    m_kernel_entry->gpgpu_ctx->device_runtime->g_total_param_size -=
        ((m_kernel_entry->get_args_aligned_size() + 255) / 256 * 256);
    m_parent_kernel->remove_child(this);
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager
        ->register_finished_kernel(m_parent_kernel->get_uid());
  }
}

CUstream_st *kernel_info_t::create_stream_cta(dim3 ctaid) {
  assert(get_default_stream_cta(ctaid));
  CUstream_st *stream = new CUstream_st();
  m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(stream);
  assert(m_cta_streams.find(ctaid) != m_cta_streams.end());
  assert(m_cta_streams[ctaid].size() >= 1);  // must have default stream
  m_cta_streams[ctaid].push_back(stream);

  return stream;
}

CUstream_st *kernel_info_t::get_default_stream_cta(dim3 ctaid) {
  if (m_cta_streams.find(ctaid) != m_cta_streams.end()) {
    assert(m_cta_streams[ctaid].size() >=
           1);  // already created, must have default stream
    return *(m_cta_streams[ctaid].begin());
  } else {
    m_cta_streams[ctaid] = std::list<CUstream_st *>();
    CUstream_st *stream = new CUstream_st();
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(
        stream);
    m_cta_streams[ctaid].push_back(stream);
    return stream;
  }
}

bool kernel_info_t::cta_has_stream(dim3 ctaid, CUstream_st *stream) {
  if (m_cta_streams.find(ctaid) == m_cta_streams.end()) return false;

  std::list<CUstream_st *> &stream_list = m_cta_streams[ctaid];
  if (std::find(stream_list.begin(), stream_list.end(), stream) ==
      stream_list.end())
    return false;
  else
    return true;
}

void kernel_info_t::print_parent_info() {
  if (m_parent_kernel) {
    printf("Parent %d: \'%s\', Block (%d, %d, %d), Thread (%d, %d, %d)\n",
           m_parent_kernel->get_uid(), m_parent_kernel->name().c_str(),
           m_parent_ctaid.x, m_parent_ctaid.y, m_parent_ctaid.z, m_parent_tid.x,
           m_parent_tid.y, m_parent_tid.z);
  }
}

void kernel_info_t::destroy_cta_streams() {
  printf("Destroy streams for kernel %d: ", get_uid());
  size_t stream_size = 0;
  for (auto s = m_cta_streams.begin(); s != m_cta_streams.end(); s++) {
    stream_size += s->second.size();
    for (auto ss = s->second.begin(); ss != s->second.end(); ss++)
      m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->destroy_stream(
          *ss);
    s->second.clear();
  }
  printf("size %lu\n", stream_size);
  m_cta_streams.clear();
}