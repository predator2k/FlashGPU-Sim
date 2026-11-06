// yasai additions to the functional simulation mode (PTX_SIM_MODE_FUNC=1).
//
// This module holds three switches that the fakeenv simulation backend uses when it records
// instruction traces without a GPU, all read once from the environment:
//   YASAI_FUNC_STATS=1   print one "yasai-func" line per kernel: CTAs, warp instructions,
//                        thread instructions, skipped FP warp instructions and wall seconds.
//   YASAI_SKIP_FP=1      do not compute the values of floating-point arithmetic (scalar FP ops,
//                        FP conversions, mma/wgmma/tcgen05.mma). Integer, predicate, address,
//                        memory and control instructions still execute exactly. Destination
//                        registers of a skipped instruction keep their previous value.
//   YASAI_CTA_SAMPLE=1   execute only the representative CTAs of each boundary class (see
//                        cta_is_representative); the other CTAs are skipped and their traces
//                        are synthesized offline from the representatives.
//   YASAI_TRACE_DIR=d    write one raw warp trace per kernel launch to d (k<seq>.ywr) plus a
//                        JSON sidecar (k<seq>.json); convert/func_to_yste.py turns it into the
//                        yasai trace entry format (.yste).
//   YASAI_TRACE_MAX_MB=n stop writing a kernel's trace after n MiB and mark it truncated.
//   YASAI_MBAR_DEBUG=1   print every operation of the functional mbarrier model below.
//   YASAI_TRACE_SRC=1    also write k<seq>.pcs.txt: executed PCs, counts and PTX source.
//
// Raw warp trace (.ywr), little-endian, one record per executed warp instruction:
//   magic "YWR1"
//   record: warp_idx u32, pc u32, ptx_opcode u16, class_code u8, flags u8 (bit0 write,
//           bit1 value present, bit2 auxiliary fields present), active_mask u32, dst_count u8,
//           src_count u8, line_count u16, reg_idx u16 x (dst_count + src_count), line u64 x
//           line_count (128 B line address with the touched 32 B sector mask in bits 0..3), value
//           u64 when flags bit1 is set (the loaded value of the first active lane of a global
//           load), and when flags bit2 is set the auxiliary fields: kind u8 (AuxKind), op u8
//           (AuxTmaOp or AuxMbarOp), parity u8, result u8, mbarrier address u32, count u32, tx
//           bytes u32. A warp instruction with auxiliary fields on several mbarriers writes one
//           record per mbarrier. A trace without bit 2 reads as before.
#ifndef YASAI_FUNC_H
#define YASAI_FUNC_H

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

class ptx_instruction;
class symbol;

namespace yasai_func {

// Environment switches, read once at first use.
struct Config {
  bool stats = false;        // YASAI_FUNC_STATS
  bool skip_fp = false;      // YASAI_SKIP_FP
  bool cta_sample = false;   // YASAI_CTA_SAMPLE
  std::string trace_dir;     // YASAI_TRACE_DIR: directory of the per-kernel warp traces; empty disables
  uint64_t trace_max_bytes = 0;  // YASAI_TRACE_MAX_MB in bytes; 0 means no limit
  // YASAI_LAUNCH_ONLY_REGEX: kernels whose mangled name matches (std::regex, ECMAScript, searched
  // anywhere in the name) are launched and recorded but none of their CTAs execute, so their
  // outputs keep the previous memory contents. Meant for kernels whose values do not matter in a
  // run, such as the random initialization of dummy weights; empty disables.
  std::string launch_only_regex;
  // YASAI_CTA_SAMPLE_REGEX: CTA sampling (as YASAI_CTA_SAMPLE) only for kernels whose mangled
  // name matches, and YASAI_CTA_SAMPLE_EXCLUDE_REGEX names kernels that always run every CTA
  // even when they match (persistent, stream-K, split-K kernels: their CTAs are not independent).
  // Meant for an allowlist of GEMM and attention kernels whose CTAs are independent and whose
  // work is an affine function of the CTA coordinates; empty disables.
  std::string cta_sample_regex;
  std::string cta_sample_exclude_regex;
};

// Returns the process-wide configuration.
const Config &config();

// Grid or block extent in three dimensions (plain values, so this module does not depend on the
// layout of kernel_info_t, which changes with the simulator's build macros).
struct Dim3 {
  unsigned x = 1;  // extent in x
  unsigned y = 1;  // extent in y
  unsigned z = 1;  // extent in z
};

// Per-kernel state of one kernel run: counters, trace file, executed-PC table, name, extents and
// the launch-only / CTA-sampling switches of the kernel. The functional mode can run kernels of
// different streams at the same time; each run owns a scope and selects it (scope_set) before
// it steps its CTAs, so kernel_begin, count_*, cta_is_representative and kernel_end act on that
// run's state.
struct KernelScope;

// A new, empty scope.
KernelScope *scope_new();

// Deletes a scope made by scope_new (not the selected one).
void scope_delete(KernelScope *s);

// The selected scope (to restore it with scope_set after running another kernel in between).
KernelScope *scope_get();

// Selects scope s; null selects the default scope.
void scope_set(KernelScope *s);

// Starts the per-kernel counters and the wall clock for the next kernel, and opens its trace
// when YASAI_TRACE_DIR is set.
void kernel_begin(const std::string &name, Dim3 grid, Dim3 block);

// Counts one executed warp instruction with active_threads active lanes.
void count_warp_inst(unsigned active_threads);

// Counts one CTA as executed (sampled == false) or as skipped by CTA sampling (sampled == true).
void count_cta(bool skipped);

// Prints the per-kernel line (when stats is on) for the kernel named name with the given grid
// and block extents, and closes the kernel's trace file.
void kernel_end(const std::string &name, Dim3 grid, Dim3 block);

// Marks the current kernel (between kernel_begin and kernel_end) as executed on the host
// (YASAI_FILL_KERNEL_HOST): its trace metadata and statistics line say host_executed, and it has
// no warp records.
void set_host_executed();

// Class codes of the trace entries (the OPCODE_CLASSES of the NVBit converter, so the replay
// treats both trace sources alike).
enum ClassCode : uint8_t {
  kCompute = 0,     // arithmetic, logic, moves and conversions
  kLoad = 1,        // global load
  kStore = 2,       // global store
  kAtomic = 3,      // global atomic or reduction
  kAsyncCopy = 4,   // cp.async global-to-shared copy
  kShared = 5,      // shared-memory access (ld/st/ldmatrix/stmatrix on shared space)
  kSync = 6,        // barriers, fences, mbarrier and async-copy waits
  kControl = 7,     // branches, calls, returns, exits
  kLocal = 8,       // local-memory access
  kConstant = 9,    // constant or parameter access
  kTma = 10,        // tensor memory accelerator copies
  kMma = 11,        // matrix multiply-accumulate (mma, wgmma, tcgen05.mma)
};

// Kinds of the auxiliary fields of a raw record. They carry what a replay needs of the operations
// that the functional model executes at once: the operands of a TMA copy and of an mbarrier
// operation, and the bulk async groups of TMA stores.
enum AuxKind : uint8_t {
  kAuxTma = 1,         // one TMA copy: op is an AuxTmaOp; the record's lines are its global side
  kAuxMbarrier = 2,    // one mbarrier operation: op is an AuxMbarOp
  kAuxBulkCommit = 3,  // cp.async.bulk.commit_group
  kAuxBulkWait = 4,    // cp.async.bulk.wait_group: count is its operand N
};

// Direction of a TMA copy.
enum AuxTmaOp : uint8_t {
  kAuxTmaLoad = 0,   // global to shared; completes tx bytes on the mbarrier at the mbarrier address
  kAuxTmaStore = 1,  // shared to global; completes through its bulk async group
};

// Operation on an mbarrier (the order of MBARRIER_OPS of the yasai trace entry format).
enum AuxMbarOp : uint8_t {
  kAuxMbarInit = 0,           // init: count is the expected arrivals per phase
  kAuxMbarArrive = 1,         // arrive: count arrivals
  kAuxMbarExpectTx = 2,       // expect_tx: tx more bytes
  kAuxMbarArriveExpectTx = 3, // arrive.expect_tx: count arrivals and tx more bytes
  kAuxMbarCompleteTx = 4,     // complete_tx: tx bytes arrived
  kAuxMbarWait = 5,           // try_wait or test_wait: result 1 when every lane found the phase done
};

// The auxiliary fields of one raw record. The lanes of one warp instruction add to one entry per
// (kind, op, mbarrier address), so a warp-wide operation gives the sum over its lanes.
struct WarpAux {
  uint8_t kind = 0;                     // AuxKind
  uint8_t op = 0;                       // AuxTmaOp or AuxMbarOp
  uint8_t parity = 0;                   // wait: the parity of the waited phase
  uint8_t result = 1;                   // wait: 1 when every lane found the phase done, else 0
  uint32_t mbar_addr = 0;               // shared-memory address of the mbarrier
  uint32_t count = 0;                   // expected arrivals, arrivals, or N of a wait_group
  uint32_t tx_bytes = 0;                // transaction bytes
  std::map<uint64_t, uint8_t> line_mask;  // TMA: 128 B line address -> touched 32 B sector mask
};

// The notes below are for the instruction implementations. Each returns at once when no trace is
// being written, and adds to the entries of the warp instruction that this thread executes.
// One lane's mbarrier operation other than a wait.
void aux_mbar(AuxMbarOp op, uint32_t addr, uint32_t count, uint32_t tx_bytes);
// One lane's phase check.
void aux_mbar_wait(uint32_t addr, uint32_t parity, bool done);
// One lane's TMA copy; aux_tma_span then gives the parts of its global side.
void aux_tma(AuxTmaOp op, uint32_t mbar_addr, uint32_t bytes);
// A part of the global side of the copy noted last by aux_tma on this thread.
void aux_tma_span(uint64_t addr, uint64_t bytes);
// One lane's cp.async.bulk.commit_group (kAuxBulkCommit) or wait_group N (kAuxBulkWait).
void aux_bulk(AuxKind kind, uint32_t n);
// Forgets the entries of this thread (before a warp instruction executes).
void aux_clear();
// The entries of the warp instruction this thread executed; they are forgotten here.
std::vector<WarpAux> aux_take();

// One executed warp instruction, filled by the functional core from the warp_inst_t and the
// PTX instruction, in plain values.
struct WarpRecord {
  uint32_t warp_idx = 0;        // linear CTA id * warps per CTA + warp id within the CTA
  uint32_t pc = 0;              // PTX program counter of the instruction
  uint16_t ptx_opcode = 0;      // opcodes.def index
  uint8_t class_code = 0;       // ClassCode
  bool is_write = false;        // store, atomic or reduction
  uint32_t active_mask = 0;     // active lanes
  uint8_t dst_count = 0;        // destination registers in reg_uid[0 .. dst_count)
  uint8_t src_count = 0;        // source registers after the destinations
  uint32_t reg_uid[64];         // PTX register symbol uids (dense indices assigned per kernel)
  bool global_mem = false;      // lane_addr holds global addresses of the active lanes
  uint64_t lane_addr[32];       // per-lane byte address of a global access
  uint32_t lane_bytes[32];      // per-lane access size in bytes
  bool has_value = false;       // value holds the loaded value of the first active lane
  uint64_t value = 0;           // see has_value
  const WarpAux *aux = nullptr; // auxiliary fields of this record; a TMA entry's lines replace lane_addr
};

// Returns true when a trace is being written for the current kernel.
bool tracing();

// With YASAI_TRACE_SRC=1, records the PTX source line of pc; kernel_end writes the executed PCs
// with their execution counts and source to k<seq>.pcs.txt next to the trace.
void note_pc_source(uint32_t pc, unsigned line, const char *text);

// Appends rec to the current kernel's trace (no-op unless tracing()).
void emit(const WarpRecord &rec);

// Returns the class code for a PTX opcode and the memory space type the access resolved to
// (memory_space_t's enum value; 0 when the instruction has no memory access).
uint8_t class_of(int ptx_opcode, int space_type, bool is_store, bool is_atomic);

// Returns true when pI computes only a floating-point value and YASAI_SKIP_FP is on, so the
// caller does not execute it. Also counts the skipped warp instruction when lane_id is 0.
bool skip_fp_value(const ptx_instruction *pI, unsigned lane_id);

// Returns true when the CTA with linear id cta_linear_id (x fastest) of a grid with extents grid
// must be executed under CTA sampling.
// Each grid dimension is split into the classes {first, second, interior, second-to-last, last};
// a CTA is a representative when every coordinate is first, second, second-to-last or last, or
// equals the lowest interior index (2). The second index in each dimension gives the per-
// dimension address stride used by the synthesis. Without YASAI_CTA_SAMPLE every CTA runs.
bool cta_is_representative(Dim3 grid, unsigned cta_linear_id);

// True when the current kernel (kernel_begin) matches YASAI_LAUNCH_ONLY_REGEX: no CTA runs.
bool launch_only();

// True when the current kernel runs only its representative CTAs (YASAI_CTA_SAMPLE, or a match
// of YASAI_CTA_SAMPLE_REGEX that YASAI_CTA_SAMPLE_EXCLUDE_REGEX does not name).
bool cta_sampled();

// Functional mbarrier model. FlashGPU-Sim keeps mbarrier state only in the timing model
// (barrier_set_t), so in PTX_SIM_MODE_FUNC=1 an mbarrier.try_wait never succeeds and every kernel
// that waits on an mbarrier (TMA pipelines, warp-specialized SM90/SM100 kernels) spins forever.
// These functions hold the state per (CTA key, shared-memory address): expected arrival count,
// pending arrivals, transaction bytes and the current phase. A phase completes when the pending
// arrivals and the transaction bytes both reach zero; the pending count then resets to the
// expected count and the phase parity flips.

// mbarrier.init: expected arrival count count, phase 0.
void mbar_init(uint64_t cta_key, uint32_t addr, uint32_t count);

// mbarrier.arrive (and cp.async.mbarrier.arrive, tcgen05.commit): count arrivals.
void mbar_arrive(uint64_t cta_key, uint32_t addr, uint32_t count);

// mbarrier.expect_tx: bytes more transaction bytes are expected in the current phase.
void mbar_expect_tx(uint64_t cta_key, uint32_t addr, int64_t bytes);

// mbarrier.complete_tx and the completion of a TMA load: bytes transaction bytes arrived.
void mbar_complete_tx(uint64_t cta_key, uint32_t addr, int64_t bytes);

// mbarrier.try_wait.parity / test_wait.parity: true when the phase with parity parity has
// completed, i.e. the current phase parity differs from parity. An address without state counts
// as completed (warned once), so a missed init cannot hang the simulation.
bool mbar_phase_done(uint64_t cta_key, uint32_t addr, uint32_t parity);

/// The current phase number of the mbarrier at addr (0 when it was not initialised): the state that
/// mbarrier.arrive returns, which try_wait and test_wait without .parity take as the phase to wait for.
uint32_t mbar_phase(uint64_t cta_key, uint32_t addr);

/// A kernel fault (trap, brkpt), as a GPU reports it (tests/gaps_m4.cu, on an RTX 5090 and a GB10): the launch
/// has returned 0, the kernel stops, what it printed reaches stdout, and from then on, in the whole process,
/// every CUDA call that touches a context returns 719 (CUDA_ERROR_LAUNCH_FAILED).
/// raise_kernel_fault prints one yasai-kernel-fault line at the first fault (the kernel, instruction and PC) and
/// sets the flag; kernel_fault reads it.
void raise_kernel_fault(const char *what, const char *kernel, unsigned long long pc, unsigned ctax, unsigned ctay,
                        unsigned ctaz, unsigned tidx, unsigned tidy, unsigned tidz, const char *source);
bool kernel_fault();

// mbarrier.inval: forget the state.
void mbar_inval(uint64_t cta_key, uint32_t addr);

// ---- Thread-block clusters in the functional mode ----------------------------------------------
// A launch with a cluster shape larger than one CTA runs cluster by cluster: all CTAs of a
// cluster are resident together and their warps are interleaved (cuda-sim.cc), so distributed
// shared memory, remote mbarriers, cluster barriers and multicast copies see one another.
//
// While a cluster executes, every shared-memory address carries the rank of the CTA that owns it
// in bits 24 and up, as on the hardware: rank r's offset o is (r << 24) | o, for the CTA's own
// variables too (the address of a shared symbol gets the executing CTA's rank, see
// ptx_thread_info::get_operand_value). CUTLASS clears bit 24 of its own mbarrier address to
// reach the leader CTA of a CTA pair, so the encoding must match. Generic addresses of the
// window are SHARED_GENERIC_START + address (cuda-sim.cc shared_to_generic). Matrix
// descriptors (wgmma, tcgen05) hold only 18 address bits and refer to the issuing CTA's memory.
// Without an executing cluster addresses are plain offsets.

// Encoded .shared::cluster address of offset in the CTA of rank rank.
inline uint64_t dsmem_encode(unsigned rank, uint64_t offset) {
  return (uint64_t(rank) << 24) | (offset & 0xFFFFFFull);
}

// Splits a cluster shared-memory address into the owner's rank and the offset.
inline void dsmem_decode(uint64_t addr, unsigned &rank, uint64_t &offset) {
  rank = unsigned((addr >> 24) & 0xFFull);
  offset = addr & 0xFFFFFFull;
}

// Rank bits of the executing CTA (0 without a cluster), to OR into an own shared address.
uint64_t own_rank_bits();

// Generic addresses follow the hardware layout: while a cluster executes, the generic shared
// window of rank r is SHARED_GENERIC_START + (r << 24), so code that derives a peer address by
// flipping bit 24 of a generic address (FlashMLA's get_peer_addr) reaches the peer CTA.

// Rank of the CTA whose warps execute now (set by the cluster executor before each CTA's round).
unsigned current_rank();
void set_current_rank(unsigned rank);

// State of a cluster that the functional executor runs; all CTAs of a cluster run on one thread.
struct ClusterState {
  unsigned size = 0;                   // CTAs in the cluster
  std::vector<void *> smem;            // rank -> the CTA's own shared memory (memory_space *)
  std::vector<uint64_t> cta_key;       // rank -> the CTA's uid (functional mbarrier key)
  std::vector<void *> cores;           // rank -> the CTA's functionalCoreSim
  std::vector<unsigned> hw_cta;        // rank -> the CTA's hardware slot (its TMEM scope)
  unsigned threads_total = 0;          // live threads of the cluster (cluster barrier count)
  unsigned barrier_arrived = 0;        // threads arrived in the current barrier.cluster phase
  unsigned barrier_phase = 0;          // completed barrier.cluster phases
};

// The cluster the calling thread executes, or null when the launch has no cluster (or between
// clusters). Per thread, as current_rank().
ClusterState *cluster();

// Starts (state non-null) or ends (null) the execution of a cluster.
void set_cluster(ClusterState *state);

// Rank of the CTA whose functional mbarrier key is cta_key in the executing cluster, or -1.
int cluster_rank_of(uint64_t cta_key);

// Rewrites (cta_key, addr) of an mbarrier in another CTA of the cluster (addr carries a rank)
// to the owner's key and plain offset; other addresses are left unchanged.
void mbar_canonical(uint64_t &cta_key, uint32_t &addr);

// ---- Small floating-point formats (FP8, FP6, FP4, E8M0 scales) -------------------------------
// Formats of the OCP microscaling specification and of PTX: e4m3 (no infinity, NaN 0x7F, max
// 448), e5m2 (IEEE-like, max 57344), e2m3 and e3m2 (FP6, max 7.5 and 28), e2m1 (FP4, max 6),
// ue8m0 (unsigned power of two 2^(e-127), NaN 0xFF).
enum MiniFloat { kE4M3, kE5M2, kE2M1, kE2M3, kE3M2, kUE8M0 };

// Value of the bit pattern bits in format f.
float minifloat_decode(MiniFloat f, unsigned bits);

// Bit pattern of x in format f. Round to nearest even unless toward_zero (or, for ue8m0,
// round_up toward +infinity). satfinite clamps out-of-range values to the largest finite value
// (e5m2 otherwise gives infinity, e4m3 NaN); NaN gives the format's NaN (0 for the FP4/FP6
// formats, which have none).
unsigned minifloat_encode(MiniFloat f, float x, bool satfinite, bool toward_zero = false,
                          bool round_up = false);

// Set when an mbarrier wait of the executing warp found its phase incomplete; the functional
// core then lets a diverged warp run its other path (simt_stack::yasai_yield).
extern thread_local bool g_wait_failed;

// Forgets every mbarrier (functional mode, after each CTA or cluster: CTA keys repeat in the
// next kernel launch, and a stale completed phase let a wait pass before the new launch's
// mbarrier.init; FlashMLA SM100 decode launched twice for h_q 128).
void mbar_reset_all();

// Grid barrier workspaces of cooperative launches (64 bytes of device memory each): a launch
// takes a free one (0 when none is free; the caller then allocates one) and the run gives it
// back when the kernel finishes. Thread-safe (host thread takes, simulation thread gives).
uint64_t grid_ws_take();
void grid_ws_give(uint64_t ws);

// Forgets the mbarriers of the CTA whose key is cta_key (when the CTA exits and its hardware
// slot, hence its key, may be reused while other CTAs keep their mbarriers).
void mbar_reset_cta(uint64_t cta_key);

// f32 to f16 bits, round to nearest even (overflow to infinity).
uint16_t minifloat_encode_f16(float x);

// Lazy PTX loading (libcuda/cuda_runtime_api.cc): loads the PTX file that defines the kernel or
// global variable name and returns true, or returns false when no unloaded file defines it.
// Null until a fat binary is registered lazily.
extern bool (*g_lazy_load_name)(const char *name);

// Mutex that serializes PTX parsing and lazy loading. The parser and the symbol tables are
// shared by the whole process, and two threads parse at once when the simulation thread loads
// a function body lazily (call_impl) while a host thread loads the piece of a kernel it is
// launching, or a module image; that corrupted the parser state (a double free in call_impl).
// Recursive, because a lazy load parses through the same entry points.
std::recursive_mutex &ptx_load_mutex();

// Function handles for function addresses held in data (indirect calls through a table, as in
// NCCL's ncclDevFuncTable). A handle is kFunctionHandleTag | index; the index is the order in
// which the symbols were first seen.
constexpr uint64_t kFunctionHandleTag = 0xfeed000000000000ull;

// Returns the handle of function symbol s, registering it on first use.
uint64_t function_handle(const symbol *s);

// Returns the function symbol of a handle, or null when value is not a handle.
const symbol *function_of(uint64_t value);

// ---- CTA-parallel functional execution (YASAI_FUNC_THREADS) -----------------------------------
// With YASAI_FUNC_THREADS=N (N > 1) the simulation thread steps the CTA groups of a kernel run on
// a pool of N host threads (yasai_func_run::step in cuda-sim.cc); the simulation thread is pool
// thread 0 and N - 1 worker threads are started at the first parallel step. N = 1 (the default)
// keeps the serial code path. The functions below are the shared state that parallel CTAs need.

// Most pool threads (the size of the per-thread tables of dyn_ptx_inst_manager and ptx-stats).
constexpr unsigned kMaxFuncThreads = 32;

// YASAI_FUNC_THREADS, clamped to [1, kMaxFuncThreads] (read once); 1 whenever the FIO window model
// is enabled (YASAI_FIO_UNORDERED_IMPORTS set), which prints one "yasai-fio-window: func_threads
// forced to 1" line when YASAI_FUNC_THREADS asked for more.
unsigned func_threads();

// Index of the calling thread in the per-thread tables: the pool index of a pool worker
// (1 .. N - 1), otherwise omp_get_thread_num() (0 outside an OpenMP region, which includes the
// simulation thread).
int thread_index();

// Runs job(i) for i in [0, n) on n pool threads, the caller as thread 0, and returns when every
// call has returned. n is at most kMaxFuncThreads; calls are serialized.
//   n: threads (1 runs job(0) on the caller); job: the work of one thread
void pool_run(unsigned n, const std::function<void(unsigned)> &job);

// Counts of one pool thread during a parallel step, added to the kernel scope's counters and the
// process instruction count by flush_thread_counts (instead of a shared counter per instruction).
struct ThreadCounts {
  uint64_t ptx_insn = 0;         // thread instructions (cuda_sim::g_ptx_sim_num_insn)
  uint64_t warp_insts = 0;       // see count_warp_inst
  uint64_t thread_insts = 0;     // see count_warp_inst
  uint64_t skipped_fp_warp = 0;  // see skip_fp_value
};

// The calling thread's counts while it executes a share of a parallel step, else null.
extern thread_local ThreadCounts *t_counts;

// Adds c to the selected scope's counters and to *ptx_insn (the process instruction count), with
// atomic additions, and clears c.
void flush_thread_counts(ThreadCounts &c, uint64_t *ptx_insn);

// Locks that make read-modify-write updates of memory atomic across pool threads: atom, red and
// multimem reductions on simulator storage and on host_window memory (which also use host
// compare-and-swap, for other processes), and the bulk and tensor reductions of TMA. A lock
// covers a 64-byte granule of addresses; granule g uses mutex g mod kRmwLocks, so a range of
// granules uses a run of consecutive mutexes, which it takes in increasing index order (all of
// them for a range of kRmwLocks granules or more).
class RmwGuard {
 public:
  static constexpr unsigned kRmwLocks = 1024;  // mutexes
  // Locks the granules of [addr, addr + bytes) (bytes 0 counts as 1).
  RmwGuard(uint64_t addr, uint64_t bytes);
  ~RmwGuard();
  RmwGuard(const RmwGuard &) = delete;
  RmwGuard &operator=(const RmwGuard &) = delete;

 private:
  unsigned m_first = 0;  // mutex index of the first granule
  unsigned m_count = 0;  // mutexes locked from m_first on, modulo kRmwLocks
};

}  // namespace yasai_func

#endif  // YASAI_FUNC_H
