#ifndef __cuda_api_object_h__
#define __cuda_api_object_h__

#include "../src/cuda-sim/yasai_func.h"
#include <functional>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "builtin_types.h"

#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "cuobjdump.h"

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

struct glbmap_entry {
  GLuint m_bufferObj;
  void *m_devPtr;
  size_t m_size;
  struct glbmap_entry *m_next;
};

typedef struct glbmap_entry glbmap_entry_t;

struct _cuda_device_id {
  _cuda_device_id(gpgpu_sim *gpu) {
    m_id = 0;
    m_next = NULL;
    m_gpgpu = gpu;
  }
  struct _cuda_device_id *next() { return m_next; }
  // yasai: the target's SM count (see gpgpu_sim_config::reported_num_shader).
  unsigned num_shader() const { return m_gpgpu->get_config().reported_num_shader(); }
  int num_devices() const {
    if (m_next == NULL)
      return 1;
    else
      return 1 + m_next->num_devices();
  }
  struct _cuda_device_id *get_device(unsigned n) {
    assert(n < (unsigned)num_devices());
    struct _cuda_device_id *p = this;
    for (unsigned i = 0; i < n; i++) p = p->m_next;
    return p;
  }
  const struct cudaDeviceProp *get_prop() const { return m_gpgpu->get_prop(); }
  unsigned get_id() const { return m_id; }

  gpgpu_sim *get_gpgpu() { return m_gpgpu; }

 private:
  unsigned m_id;
  class gpgpu_sim *m_gpgpu;
  struct _cuda_device_id *m_next;
};

struct CUctx_st {
  CUctx_st(_cuda_device_id *gpu) {
    m_gpu = gpu;
    m_binary_info.cmem = 0;
    m_binary_info.gmem = 0;
    no_of_ptx = 0;
  }

  _cuda_device_id *get_device() { return m_gpu; }

  void add_binary(symbol_table *symtab, unsigned fat_cubin_handle) {
    m_code[fat_cubin_handle] = symtab;
    m_last_fat_cubin_handle = fat_cubin_handle;
    // yasai: one fat binary can hold many PTX modules (a library built with relocatable
    // device code, e.g. NCCL); each module's table is kept so register_function finds its
    // kernels, while m_code keeps the last table per handle as before.
    bool known = false;
    for (symbol_table *t : m_all_code) known = known || (t == symtab);
    if (!known) m_all_code.push_back(symtab);
  }

  void add_ptxinfo(const char *deviceFun,
                   const struct gpgpu_ptx_sim_info &info) {
    // Try to find in last fat cubin first
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
    
    // If not found, search in all fat cubins
    if (s == NULL) {
      for (auto& pair : m_code) {
        s = pair.second->lookup(deviceFun);
        if (s != NULL) break;
      }
    }
    
    if (s == NULL) {
      fprintf(stderr, "GPGPU-Sim PTX: ERROR: cannot find deviceFun '%s' in any fat cubin\n", deviceFun);
      abort();
    }

    function_info *f = s->get_pc();
    if (f == NULL) {
      fprintf(stderr, "GPGPU-Sim PTX: ERROR: deviceFun '%s' has no function_info\n", deviceFun);
      abort();
    }
    f->set_kernel_info(info);
  }

  void add_ptxinfo(const struct gpgpu_ptx_sim_info &info) {
    m_binary_info = info;
  }

  void register_function(unsigned fat_cubin_handle, const char *hostFun,
                         const char *deviceFun) {
    if (m_code.find(fat_cubin_handle) != m_code.end()) {
      symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
      // yasai: fall back to every loaded module (see add_binary).
      for (size_t k = 0; s == NULL && k < m_all_code.size(); ++k)
        s = m_all_code[k]->lookup(deviceFun);
      if (s != NULL) {
        function_info *f = s->get_pc();
        assert(f != NULL);
        // yasai: cuModuleGetFunction registers the function_info pointer as the host key, so a
        // module image loaded again (Triton and DeepGEMM load the same image per device or per
        // call) registers the same key; the latest definition is kept.
        m_kernel_lookup[hostFun] = f;
      } else {
        // yasai: with lazy PTX loading the kernel's file is parsed at its first launch.
        if (yasai_func::g_lazy_load_name == nullptr)
          printf("Warning: cannot find deviceFun %s\n", deviceFun);
        m_kernel_lookup[hostFun] = NULL;
        m_yasai_pending_kernel[hostFun] = deviceFun;
      }
      //		assert( s != NULL );
      //		function_info *f = s->get_pc();
      //		assert( f != NULL );
      //		m_kernel_lookup[hostFun] = f;
    } else {
      // yasai: a fat binary without a PTX module for this architecture (for example a translation
      // unit built as SASS only). The kernel name is kept, so a later lookup can still find the
      // same kernel (a template instantiation is emitted in every unit that uses it) in another
      // loaded or lazily indexed PTX file, and names it when none exists.
      m_kernel_lookup[hostFun] = NULL;
      m_yasai_pending_kernel[hostFun] = deviceFun;
    }
  }

  void register_hostFun_function(const char *hostFun, function_info *f) {
    m_kernel_lookup[hostFun] = f;
  }

  // yasai: device name of a kernel registered without PTX, or "(unknown)".
  //   hostFun: host stub address the kernel was registered with
  const char *yasai_pending_name(const void *hostFun) const {
    auto p = m_yasai_pending_kernel.find(hostFun);
    return p == m_yasai_pending_kernel.end() ? "(unknown)" : p->second.c_str();
  }

  function_info *get_kernel(const char *hostFun) {
    auto i = m_kernel_lookup.find(hostFun);
    if (i == m_kernel_lookup.end()) {
      printf("Warning: cannot find hostFun %p\n", hostFun);
      abort();
    }
    // yasai: a kernel registered before its PTX file was loaded (lazy loading). The file may
    // already be loaded by an earlier kernel of the same file (layer_norm, then rms_norm from
    // the same libtorch_cuda PTX file), so the symbol is looked up whether or not this call
    // loaded it; before, the second kernel stayed NULL and its launch crashed.
    if (i->second == NULL) {
      auto p = m_yasai_pending_kernel.find(hostFun);
      if (p != m_yasai_pending_kernel.end()) {
        if (yasai_func::g_lazy_load_name != nullptr)
          yasai_func::g_lazy_load_name(p->second.c_str());
        symbol *s = NULL;
        for (size_t k = m_all_code.size(); s == NULL && k > 0; --k)
          s = m_all_code[k - 1]->lookup(p->second.c_str());
        if (s != NULL && s->get_pc() != NULL) {
          i->second = s->get_pc();
          m_yasai_pending_kernel.erase(p);
        } else if (m_yasai_reported_missing.insert(p->second).second) {
          // yasai: name the kernel once, so a cudaErrorInvalidDeviceFunction can be traced.
          printf("GPGPU-Sim PTX: no PTX loaded for kernel '%s' (host function %p)\n",
                 p->second.c_str(), hostFun);
        }  // else NULL: callers handle kernels without PTX (NCCL queries its symmetric-memory
           // kernels, which have no PTX for this architecture)
      }
    }
    return i->second;
  }

  symbol_table *get_symbol_table(unsigned fat_cubin_handle) {
    auto code_iter = m_code.find(fat_cubin_handle);
    if (code_iter != m_code.end()) {
      return code_iter->second;
    }
    return NULL;
  }

  int no_of_ptx;

 private:
  _cuda_device_id *m_gpu;  // selected gpu
  std::map<unsigned, symbol_table *>
      m_code;  // fat binary handle => global symbol table
  unsigned m_last_fat_cubin_handle;
  std::vector<symbol_table *> m_all_code;  // yasai: every module table, in load order
  // yasai: host stub -> device kernel name of kernels whose PTX was not loaded at registration.
  std::map<const void *, std::string> m_yasai_pending_kernel;
  // yasai: kernel names already reported as having no PTX (each is printed once).
  std::set<std::string> m_yasai_reported_missing;
  std::map<const void *, function_info *>
      m_kernel_lookup;  // unique id (CUDA app function address) => kernel entry
                        // point
  struct gpgpu_ptx_sim_info m_binary_info;
};

class kernel_config {
 public:
  kernel_config(dim3 GridDim, dim3 BlockDim, size_t sharedMem,
                struct CUstream_st *stream) {
    m_GridDim = GridDim;
    m_BlockDim = BlockDim;
    m_sharedMem = sharedMem;
    m_stream = stream;
  }
  kernel_config() {
    m_GridDim = dim3(-1, -1, -1);
    m_BlockDim = dim3(-1, -1, -1);
    m_sharedMem = 0;
    m_stream = NULL;
  }
  void set_arg(const void *arg, size_t size, size_t offset) {
    m_args.push_front(gpgpu_ptx_sim_arg(arg, size, offset));
  }
  dim3 grid_dim() const { return m_GridDim; }
  dim3 block_dim() const { return m_BlockDim; }
  size_t shared_mem() const { return m_sharedMem; }
  void set_grid_dim(dim3 *d) { m_GridDim = *d; }
  void set_block_dim(dim3 *d) { m_BlockDim = *d; }
  gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
  struct CUstream_st *get_stream() { return m_stream; }

 private:
  dim3 m_GridDim;
  dim3 m_BlockDim;
  size_t m_sharedMem;
  struct CUstream_st *m_stream;
  gpgpu_ptx_sim_arg_list_t m_args;
};

class cuda_runtime_api {
 public:
  cuda_runtime_api(gpgpu_context *ctx) {
    g_glbmap = NULL;
    g_active_device = 0;  // active gpu that runs the code
    gpgpu_ctx = ctx;
  }
  // global list
  std::list<cuobjdumpSection *> cuobjdumpSectionList;
  std::list<cuobjdumpSection *> libSectionList;
  std::list<kernel_config> g_cuda_launch_stack;
  std::map<int, bool> fatbin_registered;
  std::map<int, std::string> fatbinmap;
  std::map<std::string, symbol_table *> name_symtab;
  std::map<unsigned long long, size_t> g_mallocPtr_Size;
  // maps sm version number to set of filenames
  std::map<unsigned, std::set<std::string> > version_filename;
  std::string app_binary_path;
  std::map<void *, void **> pinned_memory;  // support for pinned memories added
  std::map<void *, size_t> pinned_memory_size;
  glbmap_entry_t *g_glbmap;
  int g_active_device;  // active gpu that runs the code
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // member function list

  // For SST and other potential simulator interface
  void cuobjdumpInit(const char *fn);
  void extract_code_using_cuobjdump(const char *fn);
  void extract_ptx_files_using_cuobjdump(CUctx_st *context, const char *fn);

  // For running GPGPUSim alone
  void cuobjdumpInit();
  void extract_code_using_cuobjdump();
  void extract_ptx_files_using_cuobjdump(CUctx_st *context);

  // Internal functions for the above public methods
  void cuobjdumpInit_internal(std::function<void()> ctx_extract_code_func);
  void extract_code_using_cuobjdump_internal(
      CUctx_st *context, std::string &app_binary,
      std::function<void(CUctx_st *)> ctx_extract_ptx_func);
  void extract_ptx_files_using_cuobjdump_internal(CUctx_st *context,
                                                  std::string &app_binary);

  std::list<cuobjdumpSection *> pruneSectionList(CUctx_st *context);
  std::list<cuobjdumpSection *> mergeMatchingSections(std::string identifier);
  std::list<cuobjdumpSection *> mergeSections();
  cuobjdumpELFSection *findELFSection(const std::string identifier);
  cuobjdumpPTXSection *findPTXSection(const std::string identifier);
  cuobjdumpPTXSection *findPTXSectionInList(
      std::list<cuobjdumpSection *> &sectionlist, const std::string identifier);
  void cuobjdumpRegisterFatBinary(unsigned int handle, const char *filename,
                                  CUctx_st *context);
  kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(const char *kernel_key,
                                              gpgpu_ptx_sim_arg_list_t args,
                                              struct dim3 gridDim,
                                              struct dim3 blockDim,
                                              struct CUctx_st *context);
  int load_static_globals(symbol_table *symtab, addr_t min_gaddr,
                          gpgpu_t *gpu);
  int load_constants(symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu);
};
#endif /* __cuda_api_object_h__ */
