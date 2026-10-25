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
//           bit1 value present), active_mask u32, dst_count u8, src_count u8, line_count u16,
//           reg_idx u16 x (dst_count + src_count), line u64 x line_count (128 B line address
//           with the touched 32 B sector mask in bits 0..3), value u64 when flags bit1 is set
//           (the loaded value of the first active lane of a global load).
#ifndef YASAI_FUNC_H
#define YASAI_FUNC_H

#include <cstdint>
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

// State of the cluster that the functional executor runs; one cluster at a time.
struct ClusterState {
  unsigned size = 0;                   // CTAs in the cluster
  std::vector<void *> smem;            // rank -> the CTA's own shared memory (memory_space *)
  std::vector<uint64_t> cta_key;       // rank -> the CTA's uid (functional mbarrier key)
  std::vector<void *> cores;           // rank -> the CTA's functionalCoreSim
  unsigned threads_total = 0;          // live threads of the cluster (cluster barrier count)
  unsigned barrier_arrived = 0;        // threads arrived in the current barrier.cluster phase
  unsigned barrier_phase = 0;          // completed barrier.cluster phases
};

// The executing cluster, or null when the launch has no cluster (or between clusters).
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

}  // namespace yasai_func

#endif  // YASAI_FUNC_H
