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
#include <string>

class ptx_instruction;

namespace yasai_func {

// Environment switches, read once at first use.
struct Config {
  bool stats = false;        // YASAI_FUNC_STATS
  bool skip_fp = false;      // YASAI_SKIP_FP
  bool cta_sample = false;   // YASAI_CTA_SAMPLE
  std::string trace_dir;     // YASAI_TRACE_DIR: directory of the per-kernel warp traces; empty disables
  uint64_t trace_max_bytes = 0;  // YASAI_TRACE_MAX_MB in bytes; 0 means no limit
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

}  // namespace yasai_func

#endif  // YASAI_FUNC_H
