#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include <atomic>
#include <string>
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/cuda_device_runtime.h"
#include "../src/cuda-sim/ptx-stats.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpusim_entrypoint.h"
#include "cuda_api_object.h"
#include <cassert>

class ptx_instruction;

// yasai: the PC -> instruction table with stable storage. It was a std::vector, so an append
// (assembly of a function: a kernel piece loaded by a host thread, a function body loaded lazily
// by the simulation thread inside call_impl) could reallocate it while another thread read an
// entry. Entries live in chunks of kChunk pointers that are never moved; the chunk directory has
// a fixed size. An append writes the entry and then publishes the new size (release); a reader
// that sees a PC below size() (acquire) sees its entry. Appends are serialized by the PTX load
// mutex (yasai_func::ptx_load_mutex).
class yasai_pc_table {
 public:
  static const unsigned kChunkBits = 20;                  // 1M entries per chunk
  static const size_t kChunk = size_t(1) << kChunkBits;   // entries per chunk
  static const size_t kMaxChunks = size_t(1) << 14;       // directory size: 16G entries
  yasai_pc_table() : m_size(0) {
    for (size_t c = 0; c < kMaxChunks; ++c) m_chunks[c].store(nullptr, std::memory_order_relaxed);
  }
  ~yasai_pc_table() {
    for (size_t c = 0; c < kMaxChunks; ++c) delete[] m_chunks[c].load(std::memory_order_relaxed);
  }
  yasai_pc_table(const yasai_pc_table &) = delete;
  yasai_pc_table &operator=(const yasai_pc_table &) = delete;
  // Number of entries (PCs) published.
  size_t size() const { return m_size.load(std::memory_order_acquire); }
  // Kept for the callers of the std::vector interface; storage grows by chunks.
  //   n: ignored
  void reserve(size_t n) { (void)n; }
  // Appends the instruction of the next PC (NULL for the slots inside a multi-slot instruction).
  //   p: instruction or NULL
  void push_back(ptx_instruction *p) {
    const size_t i = m_size.load(std::memory_order_relaxed);
    const size_t c = i >> kChunkBits;
    assert(c < kMaxChunks);
    ptx_instruction **chunk = m_chunks[c].load(std::memory_order_relaxed);
    if (chunk == nullptr) {
      chunk = new ptx_instruction *[kChunk]();
      m_chunks[c].store(chunk, std::memory_order_release);
    }
    chunk[i & (kChunk - 1)] = p;
    m_size.store(i + 1, std::memory_order_release);
  }
  // Instruction of PC i (i < size()).
  //   i: PC
  ptx_instruction *operator[](size_t i) const {
    return m_chunks[i >> kChunkBits].load(std::memory_order_acquire)[i & (kChunk - 1)];
  }

 private:
  std::atomic<size_t> m_size;                           // published entries
  std::atomic<ptx_instruction **> m_chunks[kMaxChunks];  // chunk directory
};

class gpgpu_context {
 public:
  gpgpu_context() {
    g_global_allfiles_symbol_table = NULL;
    sm_next_access_uid = 0;
    warp_inst_sm_next_uid = 0;
    operand_info_sm_next_uid = 1;
    kernel_info_m_next_uid = 1;
    g_num_ptx_inst_uid = 0;
    g_ptx_cta_info_uid = 1;
    symbol_sm_next_uid = 1;
    function_info_sm_next_uid = 1;
    ptx_register_allocator_enabled = false;
    ptx_register_allocator_stats = false;
    ptx_reorder_enabled = false;
    ptx_reorder_sass_guided = false;
    api = new cuda_runtime_api(this);
    ptxinfo = new ptxinfo_data(this);
    ptx_parser = new ptx_recognizer(this);
    the_gpgpusim = new GPGPUsim_ctx(this);
    func_sim = new cuda_sim(this);
    device_runtime = new cuda_device_runtime(this);
    stats = new ptx_stats(this);
  }
  // global list
  symbol_table *g_global_allfiles_symbol_table;
  const char *g_filename;
  std::atomic<unsigned> sm_next_access_uid;
  std::atomic<unsigned> warp_inst_sm_next_uid;
  unsigned operand_info_sm_next_uid;  // uid for operand_info
  unsigned kernel_info_m_next_uid;    // uid for kernel_info_t
  unsigned g_num_ptx_inst_uid;        // uid for ptx inst inside ptx_instruction
  unsigned long long g_ptx_cta_info_uid;
  unsigned symbol_sm_next_uid;  // uid for symbol
  unsigned function_info_sm_next_uid;
  yasai_pc_table s_g_pc_to_insn;  // a direct mapping from PC to instruction (yasai: stable)
  bool ptx_register_allocator_enabled;
  bool ptx_register_allocator_stats;
  bool ptx_reorder_enabled;
  bool ptx_reorder_sass_guided;
  std::string ptx_reorder_sass_ptxline_file;

  // SST related
  bool requested_synchronize = false;

  // objects pointers for each file
  cuda_runtime_api *api;
  ptxinfo_data *ptxinfo;
  ptx_recognizer *ptx_parser;
  GPGPUsim_ctx *the_gpgpusim;
  cuda_sim *func_sim;
  cuda_device_runtime *device_runtime;
  ptx_stats *stats;
  // member function list
  void synchronize();
  bool synchronize_check();
  void exit_simulation();
  void print_simulation_time();
  int gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid);
  void cuobjdumpParseBinary(unsigned int handle);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_string(const char *p,
                                                         unsigned source_num);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename(
      const char *filename);
  // yasai: compile_only runs ptxas with --compile-only (a piece that declares functions it does
  // not define, see the lazy function bodies of cuda_runtime_api.cc).
  void gpgpu_ptx_info_load_from_filename(const char *filename,
                                         const char *arch_str, bool compile_only = false);
  void gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                      unsigned source_num,
                                      unsigned sm_version = 20,
                                      int no_of_ptx = 0);
  void print_ptx_file(const char *p, unsigned source_num, const char *filename);
  class symbol_table *init_parser(const char *);
  class gpgpu_sim *gpgpu_ptx_sim_init_perf();
  void start_sim_thread(int api);
  struct _cuda_device_id *GPGPUSim_Init();
  void ptx_reg_options(option_parser_t opp);
  const ptx_instruction *pc_to_instruction(unsigned pc);
  const warp_inst_t *ptx_fetch_inst(address_type pc);
  unsigned translate_pc_to_ptxlineno(unsigned pc);
};
gpgpu_context *GPGPU_Context();

#endif /* __gpgpu_context_h__ */
