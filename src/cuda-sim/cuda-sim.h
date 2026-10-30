// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef CUDASIM_H_INCLUDED
#define CUDASIM_H_INCLUDED

#include <stdlib.h>
#include <map>
#include "yasai_func.h"
#include <string>
#include <vector>
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/shader.h"
#include "ptx_sim.h"

class gpgpu_context;
class memory_space;
class function_info;
class symbol_table;

extern const char *g_gpgpusim_version_string;
extern int g_debug_execution;

extern void print_splash();

extern void ptxinfo_opencl_addinfo(
    std::map<std::string, function_info *> &kernels);
unsigned ptx_sim_init_thread(kernel_info_t &kernel,
                             class ptx_thread_info **thread_info, int sid,
                             unsigned tid, unsigned threads_left,
                             unsigned num_threads, class core_t *core,
                             unsigned hw_cta_id, unsigned hw_warp_id,
                             gpgpu_t *gpu,
                             bool functionalSimulationMode = false);
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const class function_info *kernel);

/*!
 * This class functionally executes a kernel. It uses the basic data structures
 * and procedures in core_t
 */
// yasai: hardware CTA slots of the live functional cores (cuda-sim.cc). A slot is the SM index
// under which ptx_sim_init_thread keeps a CTA's shared memory, CTA and warp state, and the CTA
// key of tcgen05 and TMA state. Slots used to be the CTA's rank in its cluster (0 outside
// clusters), so two kernels running at the same time on different streams (NVSHMEM's internal
// stream) shared slot 0 and hit the ptx_cta_info SM index assertion. Each live core now holds a
// unique slot; released slots are reused lowest first.
unsigned yasai_acquire_cta_slot();
//   slot: a slot from yasai_acquire_cta_slot
void yasai_release_cta_slot(unsigned slot);

class functionalCoreSim : public core_t {
 public:
  functionalCoreSim(kernel_info_t *kernel, gpgpu_sim *g, unsigned warp_size)
      : core_t(g, kernel, warp_size, kernel->threads_per_cta()) {
    m_warpAtBarrier = new bool[m_warp_count];
    m_liveThreadCount = new unsigned[m_warp_count];
    m_yasai_hw_cta = yasai_acquire_cta_slot();
  }
  virtual ~functionalCoreSim() {
    yasai_free_views();
    warp_exit(0);
    delete[] m_liveThreadCount;
    delete[] m_warpAtBarrier;
    yasai_release_cta_slot(m_yasai_hw_cta);
  }
  //! executes all warps till completion
  void execute(int inst_count, unsigned ctaid_cp);
  // yasai: cluster execution (cuda_sim::gpgpu_cuda_ptx_sim_main_func). yasai_init creates the
  // threads of the next CTA with hardware CTA slot hw_cta_id (each CTA of a cluster has its own
  // slot, hence its own shared memory); yasai_round executes every warp once, releases warps
  // whose barrier completed, and returns true while a thread of the CTA is live.
  void yasai_init(unsigned ctaid_cp, unsigned hw_cta_id);
  bool yasai_round();
  // yasai: joins the CTA to the executing cluster as rank rank; its threads then address shared
  // memory through a cluster view (distributed shared memory).
  void yasai_join_cluster(yasai_func::ClusterState *cluster, unsigned rank);
  // yasai: live threads of the CTA.
  unsigned yasai_live_threads() const;
  // yasai: prints, for every warp with a live thread, the PTX line its first live thread is at
  // (hang diagnosis, YASAI_HANG_ROUNDS).
  void yasai_dump_warps(FILE *out) const;
  // yasai: the CTA's own shared memory (before joining a cluster) and the uid of its CTA.
  memory_space *yasai_shared_memory() const;
  uint64_t yasai_cta_key() const;
  // yasai: the hardware CTA slot the core holds (yasai_acquire_cta_slot), which is the cta_id of
  // its TMEM scope.
  unsigned yasai_hw_cta() const { return m_yasai_hw_cta; }
  virtual void warp_exit(unsigned warp_id);
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const {
    return (m_warpAtBarrier[warp_id] || !(m_liveThreadCount[warp_id] > 0));
  }

 private:
  void executeWarp(unsigned, bool &, bool &);
  // yasai: writes the trace record of the warp instruction inst of warp w (YASAI_TRACE_DIR).
  void yasai_emit_warp(const warp_inst_t &inst, unsigned w);
  // yasai: CTA-wide and named-barrier handling of a bar instruction of warp w.
  void yasai_barrier(const warp_inst_t &inst, unsigned w);
  // yasai: barrier id a warp waits on (-1 when none), and threads arrived per barrier id.
  std::vector<int> m_yasai_wait_bar;
  std::map<unsigned, unsigned> m_yasai_bar_arrived;
  // yasai: lanes of each warp that reached a named barrier while other lanes of the warp were
  // on another divergent path (see yasai_barrier).
  std::vector<uint32_t> m_yasai_bar_lanes;
  // yasai: barrier.cluster arrive/wait of warp w (inst); see yasai_barrier.
  void yasai_cluster_barrier(const warp_inst_t &inst, unsigned w, bool arrive);
  // yasai: deletes the cluster shared-memory views.
  void yasai_free_views();
  // yasai: releases warps waiting on barrier.cluster whose phase completed.
  void yasai_release_cluster_waiters();
  unsigned m_yasai_hw_cta = 0;                          // hardware CTA slot (yasai_init)
  yasai_func::ClusterState *m_yasai_cluster = nullptr;  // executing cluster, or null
  unsigned m_yasai_rank = 0;                            // rank in the cluster
  yasai_func::ClusterState m_yasai_own_cluster;         // barrier.cluster state without a cluster
  std::vector<unsigned> m_yasai_cbar_phase;  // per warp: barrier.cluster phase it arrived in
  std::vector<char> m_yasai_cbar_arrived;    // per warp: arrived and not yet waited
  std::vector<memory_space *> m_yasai_views; // cluster shared-memory views (owned)
  // initializes threads in the CTA block which we are executing
  void initializeCTA(unsigned ctaid_cp);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) {
    if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
      m_liveThreadCount[tid / m_warp_size]--;
    }
  }

  // lunches the stack and set the threads count
  void createWarp(unsigned warpId);

  // each warp live thread count and barrier indicator
  unsigned *m_liveThreadCount;
  bool *m_warpAtBarrier;
};

class cuda_sim;

/*!
 * yasai: functional execution of one kernel launch as a sequence of rounds, so that the
 * simulation thread can run several kernels (of different streams) at the same time and the
 * CTAs of one kernel can be co-resident.
 *
 * The run keeps a set of CTA groups active at once; a group is one CTA, or the CTAs of one
 * cluster. step() executes one round (every warp of every active CTA once), retires finished
 * groups and starts the next CTAs in their place. The number of active groups is
 *   * all of them for a cooperative launch (the grid fits the co-resident limit, checked at
 *     launch; CTA sampling is off because grid.sync counts every CTA);
 *   * the co-resident limit (the reported SM count x CTAs per SM for this kernel, from the
 *     device properties the application sees, not from the functional cores; divided by
 *     the cluster size) when CTA interleaving is on: YASAI_CTA_INTERLEAVE=1, cluster
 *     launches included;
 *   * one otherwise (CTA-at-a-time, the former order).
 * A kernel that launches with clusters, except cooperatively without tensor memory, is
 * exclusive: no other kernel runs beside it, and without CTA interleaving it runs with one
 * group, resetting tensor memory and mbarriers after each cluster. Tensor memory is kept
 * per CTA slot, so a TMEM kernel without clusters interleaves like any other kernel; each CTA
 * has its own 512 columns (on hardware the CTAs of one SM share them, and tcgen05.alloc waits
 * while they are taken). Checkpointing
 * (checkpoint_option) uses the former loop, run to completion by the constructor.
 * Each active CTA's core holds its own hardware CTA slot from the pool
 * (yasai_acquire_cta_slot: shared memory, CTA, warp, mbarrier, tcgen05 and local memory state),
 * shared by all concurrently running kernels; %smid is the CTA's virtual SM, its co-resident
 * placement round-robin over the reported SM count.
 */
class yasai_func_run {
 public:
  /// Prepares the kernel (PDOM analysis, occupancy, per-kernel statistics) and starts its first
  /// CTAs.
  /// @param sim Functional simulator of the process.
  /// @param kernel Kernel launch.
  /// @param openCL True for an OpenCL launch (its exit is not registered with the streams).
  yasai_func_run(cuda_sim *sim, kernel_info_t &kernel, bool openCL);
  ~yasai_func_run();
  /// Executes one round.
  /// @return True when the kernel has finished (its completion is then registered).
  bool step();
  /// Whether no other kernel may run beside this one (see the class comment).
  bool exclusive() const { return m_exclusive; }
  /// Whether a launch of kernel would be exclusive.
  static bool is_exclusive(kernel_info_t &kernel);
  /// The kernel launch.
  kernel_info_t &kernel() const { return m_kernel; }

 private:
  /// One active CTA or cluster.
  struct group {
    std::vector<functionalCoreSim *> cores;  // CTAs (cluster rank order)
    yasai_func::ClusterState state;          // cluster state (clusters only)
    unsigned long rounds = 0;                // rounds executed (YASAI_HANG_ROUNDS)
  };
  /// Starts the next CTA (or cluster) in g, the group at co-resident position pos; false when
  /// none is left.
  bool refill(group &g, unsigned pos);
  /// Virtual SM id of rank r of a group at co-resident position pos running CTA cta.
  unsigned virtual_smid(unsigned pos, unsigned r, unsigned cta) const;
  /// Deletes the CTAs of g and frees their slots.
  void retire(group &g);
  /// Registers the kernel's completion and prints the per-kernel lines.
  void finish();
  /// The former CTA-at-a-time loop with checkpoint support, run to completion.
  void run_checkpointed();

  cuda_sim *m_sim;
  kernel_info_t &m_kernel;
  bool m_openCL;
  std::string m_name;             // kernel name (the kernel object may be released at the end)
  yasai_func::Dim3 m_grid, m_block;  // extents for the statistics
  yasai_func::KernelScope *m_scope;  // per-kernel state of yasai_func
  unsigned m_csize = 1;           // CTAs per group (cluster size)
  yasai_func::Dim3 m_ncl;         // grid of clusters (clusters only)
  unsigned m_ncl_total = 0;       // clusters in the grid
  unsigned m_next_cluster = 0;    // next cluster to start (clusters only)
  bool m_coop = false;            // cooperative launch
  bool m_exclusive = false;       // see exclusive()
  bool m_uses_tmem = false;       // the kernel allocates tensor memory
  bool m_done = false;            // finish() has run
  unsigned m_warp_size = 32;
  unsigned m_sms = 1;             // SM count the application sees (virtual %smid)
  bool m_interleaved = false;     // more than one group is active at once
  std::vector<group *> m_groups;  // active groups (null: slot drained)
};

#define RECONVERGE_RETURN_PC ((address_type) - 2)
#define NO_BRANCH_DIVERGENCE ((address_type) - 1)
address_type get_return_pc(void *thd);
const char *get_ptxinfo_kname();
void print_ptxinfo();
void clear_ptxinfo();
struct gpgpu_ptx_sim_info get_ptxinfo();

class gpgpu_recon_t;
struct rec_pts {
  gpgpu_recon_t *s_kernel_recon_points;
  int s_num_recon;
};

class cuda_sim {
 public:
  cuda_sim(gpgpu_context *ctx) {
    g_ptx_sim_num_insn = 0;
    g_ptx_kernel_count =
        -1;  // used for classification stat collection purposes
    gpgpu_param_num_shaders = 0;
    g_cuda_launch_blocking = false;
    g_inst_classification_stat = NULL;
    g_inst_op_classification_stat = NULL;
    g_assemble_code_next_pc = 0;
    g_debug_thread_uid = 0;
    g_override_embedded_ptx = false;
    ptx_tex_regs = NULL;
    g_ptx_thread_info_delete_count = 0;
    g_ptx_thread_info_uid_next = 1;
    g_debug_pc = 0xBEEF1518;
    gpgpu_ctx = ctx;
  }
  // global variables
  char *opcode_latency_int;
  char *opcode_latency_fp;
  char *opcode_latency_dp;
  char *opcode_latency_sfu;
  unsigned opcode_latency_ex2;
  unsigned opcode_latency_predicate;
  unsigned opcode_latency_int_logic;
  unsigned opcode_initiation_int_logic;
  bool opcode_f32_minmax_use_int;
  unsigned opcode_latency_f32x2;
  unsigned opcode_initiation_f32x2;
  unsigned opcode_latency_cvt_f16x2_f32;
  unsigned opcode_initiation_cvt_f16x2_f32;
  char *opcode_latency_tensor;
  char *opcode_latency_wgmma_ss;
  char *opcode_latency_wgmma_rs;
  char *opcode_completion_wgmma_ss;
  char *opcode_completion_wgmma_rs;
  char *opcode_completion_wgmma_int_ss;
  char *opcode_completion_wgmma_int_rs;
  char *opcode_compute_throughput_wgmma;
  char *opcode_latency_tma;
  char *opcode_latency_cp_async;
  char *opcode_latency_cp_async_commit;
  char *opcode_latency_cp_async_wait;
  char *opcode_latency_tensormap;
  char *opcode_initiation_int;
  char *opcode_initiation_fp;
  char *opcode_initiation_dp;
  char *opcode_initiation_sfu;
  char *opcode_initiation_tensor;
  char *opcode_initiation_wgmma_ss;
  char *opcode_initiation_wgmma_rs;
  char *opcode_initiation_tma;
  char *opcode_initiation_cp_async;
  char *opcode_initiation_cp_async_commit;
  char *opcode_initiation_cp_async_wait;
  char *opcode_initiation_tensormap;
  int cp_count;
  int cp_cta_resume;
  int g_ptxinfo_error_detected;
  unsigned g_ptx_sim_num_insn;
  char *cdp_latency_str;
  int g_ptx_kernel_count;  // used for classification stat collection purposes
  std::map<const void *, std::string>
      g_global_name_lookup;  // indexed by hostVar
  std::map<const void *, std::string>
      g_const_name_lookup;  // indexed by hostVar
  int g_ptx_sim_mode;  // if non-zero run functional simulation only (i.e., no
                       // notion of a clock cycle)
  unsigned gpgpu_param_num_shaders;
  class std::map<function_info *, rec_pts> g_rpts;
  bool g_cuda_launch_blocking;
  void **g_inst_classification_stat;
  void **g_inst_op_classification_stat;
  std::set<std::string> g_globals;
  std::set<std::string> g_constants;
  std::map<unsigned, function_info *> g_pc_to_finfo;
  int gpgpu_ptx_instruction_classification;
  unsigned cdp_latency[5];
  unsigned g_assemble_code_next_pc;
  int g_debug_thread_uid;
  bool g_override_embedded_ptx;
  std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;
  ptx_reg_t *ptx_tex_regs;
  unsigned g_ptx_thread_info_delete_count;
  unsigned g_ptx_thread_info_uid_next;
  addr_t g_debug_pc;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // global functions
  void ptx_opcocde_latency_options(option_parser_t opp);
  void gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel, bool openCL = false);
  int gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid);
  void init_inst_classification_stat();
  kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                                gpgpu_ptx_sim_arg_list_t args,
                                                struct dim3 gridDim,
                                                struct dim3 blockDim,
                                                gpgpu_t *gpu);
  void gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                              const char *deviceName,
                                              size_t size);
  void gpgpu_ptx_sim_register_const_variable(void *, const char *deviceName,
                                             size_t size);
  void read_sim_environment_variables();
  void set_param_gpgpu_num_shaders(int num_shaders);
  struct rec_pts find_reconvergence_points(function_info *finfo);
  address_type get_converge_point(address_type pc);
  void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                   size_t count, size_t offset, int to,
                                   gpgpu_t *gpu);
  /// yasai: device address and size of the variable that the host registered as hostVar
  /// (__cudaRegisterVar), for cudaGetSymbolAddress / cudaGetSymbolSize; loads its PTX file when
  /// it is not parsed yet. False when hostVar names no registered variable.
  bool yasai_symbol_address(const void *hostVar, unsigned long long *addr, size_t *size);
  void ptx_print_insn(address_type pc, FILE *fp);
  std::string ptx_get_insn_str(address_type pc);
  template <int activate_level>
  bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc);
};

#endif
