// yasai additions to the functional simulation mode: statistics, FP value skipping and CTA
// sampling. See yasai_func.h for the environment switches.
#include "yasai_func.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

#include "../abstract_hardware_model.h"

#include "opcodes.h"
#include "ptx_ir.h"
#include "ptx_parser.h"
#include "ptx.tab.h"

namespace yasai_func {

namespace {

// Per-kernel counters; the functional mode runs one kernel at a time on one thread.
struct KernelCounters {
  uint64_t warp_insts = 0;        // executed warp instructions
  uint64_t thread_insts = 0;      // executed thread instructions (sum of active lanes)
  uint64_t skipped_fp_warp = 0;   // warp instructions whose FP value computation was skipped
  uint64_t ctas_run = 0;          // CTAs executed
  uint64_t ctas_skipped = 0;      // CTAs skipped by CTA sampling
  std::chrono::steady_clock::time_point start;  // wall clock at kernel_begin
};

KernelCounters g_counters;

// Trace state of the current kernel.
struct TraceState {
  FILE *file = nullptr;                          // open .ywr file, or null when not tracing
  std::string base;                              // path without extension
  uint64_t bytes = 0;                            // bytes written so far
  bool truncated = false;                        // YASAI_TRACE_MAX_MB was reached
  uint64_t records = 0;                          // records written
  std::unordered_map<uint32_t, uint16_t> reg_idx;  // PTX register uid -> dense index
  std::vector<char> buf;                         // pending output
};

TraceState g_trace;
// Sequence number of the kernel launches of this process (names the trace files).
uint64_t g_kernel_seq = 0;
// Executed PCs of the current kernel with their count, PTX line and source (YASAI_TRACE_SRC).
std::map<uint32_t, std::pair<uint64_t, std::string>> g_pc_src;

// Kernel name and extents for the sidecar.
std::string g_name;
Dim3 g_grid, g_block;

// Appends the little-endian bytes of v to the pending output.
template <typename T>
void put(const T &v) {
  const char *p = reinterpret_cast<const char *>(&v);
  g_trace.buf.insert(g_trace.buf.end(), p, p + sizeof(T));
}

// Writes the pending output to the trace file.
void flush_trace() {
  if (g_trace.file && !g_trace.buf.empty())
    std::fwrite(g_trace.buf.data(), 1, g_trace.buf.size(), g_trace.file);
  g_trace.bytes += g_trace.buf.size();
  g_trace.buf.clear();
}

// Reads a boolean environment switch: set and not "0".
bool env_flag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && std::strcmp(v, "0") != 0;
}

// Returns true for PTX data types that hold floating-point values.
bool is_float_type(int t) {
  switch (t) {
    case F16_TYPE:
    case F16X2_TYPE:
    case BF16_TYPE:
    case F32_TYPE:
    case F32X2_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
    case TF32_TYPE:
    case E4M3_TYPE:
    case E5M2_TYPE:
      return true;
    default:
      return false;
  }
}

// Returns the per-dimension sampling class index of coordinate c in a dimension of extent n:
// true when c is one of the indices 0, 1, 2, n-2, n-1 (the representatives).
bool coord_is_representative(unsigned c, unsigned n) {
  return c <= 2 || c + 2 >= n;
}

}  // namespace

const Config &config() {
  static const Config cfg = [] {
    Config c;
    c.stats = env_flag("YASAI_FUNC_STATS");
    c.skip_fp = env_flag("YASAI_SKIP_FP");
    c.cta_sample = env_flag("YASAI_CTA_SAMPLE");
    if (const char *d = std::getenv("YASAI_TRACE_DIR")) c.trace_dir = d;
    if (const char *m = std::getenv("YASAI_TRACE_MAX_MB"))
      c.trace_max_bytes = std::strtoull(m, nullptr, 10) << 20;
    return c;
  }();
  return cfg;
}

void kernel_begin(const std::string &name, Dim3 grid, Dim3 block) {
  g_counters = KernelCounters();
  g_name = name;
  g_grid = grid;
  g_block = block;
  g_trace = TraceState();
  if (!config().trace_dir.empty()) {
    char tmp[64];
    std::snprintf(tmp, sizeof(tmp), "/k%06llu", (unsigned long long)g_kernel_seq);
    g_trace.base = config().trace_dir + tmp;
    g_trace.file = std::fopen((g_trace.base + ".ywr").c_str(), "wb");
    if (g_trace.file) std::fwrite("YWR1", 1, 4, g_trace.file);
  }
  g_kernel_seq++;
  g_counters.start = std::chrono::steady_clock::now();
}

bool tracing() { return g_trace.file != nullptr && !g_trace.truncated; }

namespace {

// Writes the executed PCs recorded by note_pc_source to <trace base>.pcs.txt.
void dump_pcs() {
  if (g_pc_src.empty() || g_trace.base.empty()) return;
  FILE *pcs = std::fopen((g_trace.base + ".pcs.txt").c_str(), "w");
  if (!pcs) return;
  for (const auto &kv : g_pc_src)
    std::fprintf(pcs, "0x%x\t%llu\t%s\n", kv.first, (unsigned long long)kv.second.first,
                 kv.second.second.c_str());
  std::fclose(pcs);
}

}  // namespace

void note_pc_source(uint32_t pc, unsigned line, const char *text) {
  static const bool on = env_flag("YASAI_TRACE_SRC");
  if (!on) return;
  auto &e = g_pc_src[pc];
  if (e.first++ == 0) e.second = std::to_string(line) + "\t" + (text ? text : "");
}

void emit(const WarpRecord &rec) {
  if (!tracing()) return;
  // Coalesce the active lanes of a global access into 128 B lines with 32 B sector masks.
  uint64_t lines[64];
  unsigned nlines = 0;
  if (rec.global_mem) {
    for (unsigned l = 0; l < 32; l++) {
      if (!(rec.active_mask >> l & 1u)) continue;
      const uint64_t a = rec.lane_addr[l];
      const uint64_t last = a + (rec.lane_bytes[l] ? rec.lane_bytes[l] : 1) - 1;
      for (uint64_t s = a >> 5; s <= last >> 5; s++) {
        const uint64_t line = (s >> 2) << 7;
        const uint64_t bit = 1ull << (s & 3);
        unsigned j = 0;
        while (j < nlines && (lines[j] & ~0x7full) != line) j++;
        if (j == nlines) {
          if (nlines == 64) continue;
          lines[nlines++] = line;
        }
        lines[j] |= bit;
      }
    }
  }
  put(rec.warp_idx);
  put(rec.pc);
  put(rec.ptx_opcode);
  put(rec.class_code);
  const uint8_t flags = uint8_t((rec.is_write ? 1 : 0) | (rec.has_value ? 2 : 0));
  put(flags);
  put(rec.active_mask);
  put(rec.dst_count);
  put(rec.src_count);
  put(uint16_t(nlines));
  for (unsigned r = 0; r < unsigned(rec.dst_count) + rec.src_count; r++) {
    auto it = g_trace.reg_idx.find(rec.reg_uid[r]);
    uint16_t idx;
    if (it == g_trace.reg_idx.end()) {
      idx = uint16_t(g_trace.reg_idx.size());
      g_trace.reg_idx.emplace(rec.reg_uid[r], idx);
    } else {
      idx = it->second;
    }
    put(idx);
  }
  for (unsigned j = 0; j < nlines; j++) put(lines[j]);
  if (rec.has_value) put(rec.value);
  g_trace.records++;
  if (g_trace.buf.size() >= (1u << 22)) flush_trace();
  if (config().trace_max_bytes && g_trace.bytes + g_trace.buf.size() >= config().trace_max_bytes) {
    flush_trace();
    std::fflush(g_trace.file);
    g_trace.truncated = true;
    dump_pcs();  // a kernel that never ends still leaves its executed PCs
  }
}

uint8_t class_of(int opc, int space, bool is_store, bool is_atomic) {
  switch (opc) {
    case MMA_OP:
    case TENSOR_MMA_OP:
    case WGMMA_MMA_ASYNC_OP:
    case WGMMA_MMA_ASYNC_SP_OP:
    case TCGEN05_MMA_OP:
      return kMma;
    case TMA_OP:
    case TMA_PREFETCH_OP:
      return kTma;
    case BAR_OP:
    case MEMBAR_OP:
    case FENCE_OP:
    case MBAR_OP:
    case CP_ASYNC_WAIT_OP:
    case CP_ASYNC_COMMIT_OP:
    case CP_ASYNC_MBARRIER_ARRIVE_OP:
    case WGMMA_FENCE_OP:
    case WGMMA_COMMIT_GROUP_OP:
    case WGMMA_WAIT_GROUP_OP:
      return kSync;
    case BRA_OP:
    case BRX_OP:
    case CALL_OP:
    case CALLP_OP:
    case RET_OP:
    case RETP_OP:
    case EXIT_OP:
    case TRAP_OP:
      return kControl;
    case LDMATRIX_OP:
    case STMATRIX_OP:
      return kShared;
    case CP_ASYNC_OP:
      return kAsyncCopy;
    default:
      break;
  }
  switch (space) {
    case 0:
      return kCompute;
    case global_space:
    case generic_space:
      if (is_atomic || opc == RED_OP || opc == ATOM_OP) return kAtomic;
      return is_store ? kStore : kLoad;
    case shared_space:
    case sstarr_space:
      return kShared;
    case local_space:
    case param_space_local:
      return kLocal;
    case const_space:
    case param_space_kernel:
    case param_space_unclassified:
    case tex_space:
      return kConstant;
    default:
      return kCompute;
  }
}

void count_warp_inst(unsigned active_threads) {
  g_counters.warp_insts++;
  g_counters.thread_insts += active_threads;
}

void count_cta(bool skipped) {
  if (skipped)
    g_counters.ctas_skipped++;
  else
    g_counters.ctas_run++;
}

void kernel_end(const std::string &name, Dim3 g, Dim3 b) {
  const double trace_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                          g_counters.start)
                                .count();
  if (g_trace.file) {
    flush_trace();
    std::fclose(g_trace.file);
    FILE *meta = std::fopen((g_trace.base + ".json").c_str(), "w");
    if (meta) {
      std::fprintf(meta,
                   "{\"kernel\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], "
                   "\"warps_per_cta\": %u, \"records\": %llu, \"bytes\": %llu, "
                   "\"truncated\": %s, \"registers\": %zu, \"warp_insts\": %llu, "
                   "\"ctas_run\": %llu, \"ctas_skipped\": %llu, \"cta_sample\": %s, "
                   "\"skip_fp\": %s, \"secs\": %.6f}\n",
                   name.c_str(), g.x, g.y, g.z, b.x, b.y, b.z, (b.x * b.y * b.z + 31) / 32,
                   (unsigned long long)g_trace.records,
                   (unsigned long long)(g_trace.bytes + 4), g_trace.truncated ? "true" : "false",
                   g_trace.reg_idx.size(), (unsigned long long)g_counters.warp_insts,
                   (unsigned long long)g_counters.ctas_run,
                   (unsigned long long)g_counters.ctas_skipped,
                   config().cta_sample ? "true" : "false", config().skip_fp ? "true" : "false",
                   trace_secs);
      std::fclose(meta);
    }
    g_trace.file = nullptr;
    dump_pcs();
  }
  g_pc_src.clear();
  if (!config().stats) return;
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                    g_counters.start)
                          .count();
  std::printf(
      "yasai-func kernel=%s grid=%u,%u,%u block=%u,%u,%u ctas_run=%llu ctas_skipped=%llu "
      "warp_insts=%llu thread_insts=%llu skipped_fp_warp=%llu secs=%.6f warp_inst_per_s=%.1f "
      "skip_fp=%d cta_sample=%d\n",
      name.c_str(), g.x, g.y, g.z, b.x, b.y, b.z,
      (unsigned long long)g_counters.ctas_run, (unsigned long long)g_counters.ctas_skipped,
      (unsigned long long)g_counters.warp_insts, (unsigned long long)g_counters.thread_insts,
      (unsigned long long)g_counters.skipped_fp_warp, secs,
      secs > 0 ? g_counters.warp_insts / secs : 0.0, (int)config().skip_fp,
      (int)config().cta_sample);
  std::fflush(stdout);
}

bool skip_fp_value(const ptx_instruction *pI, unsigned lane_id) {
  if (!config().skip_fp) return false;
  bool skip = false;
  switch (pI->get_opcode()) {
    // Matrix multiply-accumulate: the result is a value only.
    case MMA_OP:
    case TENSOR_MMA_OP:
    case WGMMA_MMA_ASYNC_OP:
    case WGMMA_MMA_ASYNC_SP_OP:
    case TCGEN05_MMA_OP:
      skip = true;
      break;
    // Transcendental and reciprocal ops exist only for FP types.
    case EX2_OP:
    case LG2_OP:
    case RCP_OP:
    case RSQRT_OP:
    case SQRT_OP:
    case SIN_OP:
    case COS_OP:
    case TANH_OP:
      skip = true;
      break;
    // Arithmetic that is FP only when the instruction type is FP.
    case ADD_OP:
    case SUB_OP:
    case MUL_OP:
    case FMA_OP:
    case MAD_OP:
    case DIV_OP:
    case ABS_OP:
    case NEG_OP:
    case MIN_OP:
    case MAX_OP:
      skip = is_float_type(pI->get_type());
      break;
    // A conversion between two FP types changes only a value; FP-to-integer stays exact
    // because an index may come from it.
    case CVT_OP:
      skip = is_float_type(pI->get_type()) && is_float_type(pI->get_type2());
      break;
    default:
      skip = false;
  }
  if (skip && lane_id == 0) g_counters.skipped_fp_warp++;
  return skip;
}

bool cta_is_representative(Dim3 g, unsigned cta_linear_id) {
  if (!config().cta_sample) return true;
  const unsigned x = cta_linear_id % g.x;
  const unsigned y = (cta_linear_id / g.x) % g.y;
  const unsigned z = cta_linear_id / (g.x * g.y);
  return coord_is_representative(x, g.x) && coord_is_representative(y, g.y) &&
         coord_is_representative(z, g.z);
}

namespace {

// State of one functional mbarrier.
struct MbarState {
  uint32_t expected = 0;   // arrivals per phase (from mbarrier.init)
  int64_t pending = 0;     // arrivals still missing in the current phase
  int64_t tx = 0;          // transaction bytes still missing in the current phase
  uint32_t phase = 0;      // current phase number
};

// All functional mbarriers, keyed by (CTA key, shared-memory address).
std::map<std::pair<uint64_t, uint32_t>, MbarState> g_mbars;

// True when YASAI_MBAR_DEBUG is set: every functional mbarrier operation is printed.
bool mbar_debug() {
  static const bool on = env_flag("YASAI_MBAR_DEBUG");
  return on;
}

// Prints one functional mbarrier operation when mbar_debug() is on.
void mbar_log(const char *what, uint64_t cta_key, uint32_t addr, int64_t value,
              const MbarState *m) {
  if (!mbar_debug()) return;
  if (m)
    std::printf("yasai-mbar %s cta=%llu addr=0x%x value=%lld pending=%lld tx=%lld phase=%u\n",
                what, (unsigned long long)cta_key, addr, (long long)value, (long long)m->pending,
                (long long)m->tx, m->phase);
  else
    std::printf("yasai-mbar %s cta=%llu addr=0x%x value=%lld unknown\n", what,
                (unsigned long long)cta_key, addr, (long long)value);
}

// Completes the current phase of m when nothing is pending.
void mbar_try_advance(MbarState &m) {
  if (m.pending <= 0 && m.tx == 0) {
    m.phase++;
    m.pending = m.expected;
  }
}

}  // namespace

namespace {
ClusterState *g_cluster = nullptr;  // see cluster()
}  // namespace

ClusterState *cluster() { return g_cluster; }

bool (*g_lazy_load_name)(const char *name) = nullptr;

namespace {
unsigned g_current_rank = 0;  // see current_rank()
}  // namespace

unsigned current_rank() { return g_current_rank; }

void set_current_rank(unsigned rank) { g_current_rank = rank; }

uint64_t own_rank_bits() { return g_cluster != nullptr ? uint64_t(g_current_rank) << 24 : 0; }

namespace {
// Exponent bits, mantissa bits, bias and largest finite value of a format.
struct MiniFloatSpec {
  int ebits, mbits, bias;
  float maxv;
  bool has_sign;
};
MiniFloatSpec spec(MiniFloat f) {
  switch (f) {
    case kE4M3: return {4, 3, 7, 448.0f, true};
    case kE5M2: return {5, 2, 15, 57344.0f, true};
    case kE2M1: return {2, 1, 1, 6.0f, true};
    case kE2M3: return {2, 3, 1, 7.5f, true};
    case kE3M2: return {3, 2, 3, 28.0f, true};
    default: return {8, 0, 127, 0.0f, false};
  }
}
}  // namespace

float minifloat_decode(MiniFloat f, unsigned bits) {
  if (f == kUE8M0) {
    bits &= 0xFF;
    if (bits == 0xFF) return std::nanf("");
    return std::ldexp(1.0f, int(bits) - 127);
  }
  const MiniFloatSpec s = spec(f);
  const unsigned width = 1 + s.ebits + s.mbits;
  bits &= (1u << width) - 1;
  const bool neg = (bits >> (width - 1)) & 1u;
  const unsigned e = (bits >> s.mbits) & ((1u << s.ebits) - 1);
  const unsigned m = bits & ((1u << s.mbits) - 1);
  float v;
  if (f == kE4M3 && e == 15 && m == 7) return std::nanf("");
  if (f == kE5M2 && e == 31) return m ? std::nanf("") : (neg ? -INFINITY : INFINITY);
  if (e == 0)
    v = std::ldexp(float(m), 1 - s.bias - s.mbits);
  else
    v = std::ldexp(1.0f + float(m) / float(1u << s.mbits), int(e) - s.bias);
  return neg ? -v : v;
}

uint16_t minifloat_encode_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int e = int((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF) return uint16_t(sign | 0x7C00 | (m ? 0x200 : 0));
  if (e >= 31) return uint16_t(sign | 0x7C00);
  if (e <= 0) {
    if (e < -10) return uint16_t(sign);
    m |= 0x800000;
    const int shift = 14 - e;
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1))) r++;
    return uint16_t(sign | r);
  }
  uint32_t r = (uint32_t(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
  return uint16_t(sign | r);
}

unsigned minifloat_encode(MiniFloat f, float x, bool satfinite, bool toward_zero, bool round_up) {
  if (f == kUE8M0) {
    if (std::isnan(x) || x <= 0.0f) return std::isnan(x) ? 0xFF : 0;
    int e;
    const float m = std::frexp(x, &e);  // x = m * 2^e, m in [0.5, 1)
    int exp2 = e - 1;                    // x = (2m) * 2^(e-1), 2m in [1, 2)
    if (round_up && m != 0.5f) exp2 += 1;
    if (!round_up && !toward_zero && 2.0f * m >= 1.5f) exp2 += 1;
    int code = exp2 + 127;
    if (code < 0) code = 0;
    if (code > 254) code = satfinite ? 254 : 0xFF;
    return unsigned(code);
  }
  const MiniFloatSpec s = spec(f);
  const unsigned width = 1 + s.ebits + s.mbits;
  const unsigned sign = std::signbit(x) ? 1u << (width - 1) : 0u;
  if (std::isnan(x)) {
    if (f == kE4M3) return 0x7F;
    if (f == kE5M2) return 0x7F;
    return 0;
  }
  float a = std::fabs(x);
  const int min_exp = 1 - s.bias;  // exponent of the smallest normal
  // value on the grid: quantum of a's binade
  int e = a > 0.0f ? int(std::floor(std::log2(a))) : min_exp;
  if (e < min_exp) e = min_exp;
  const float q = std::ldexp(1.0f, e - s.mbits);
  float n = a / q;
  float r = toward_zero ? std::floor(n) : std::nearbyint(n);  // nearbyint: ties to even
  float v = r * q;
  if (v > s.maxv || std::isinf(a)) {
    if (satfinite || f == kE2M1 || f == kE2M3 || f == kE3M2) v = s.maxv;
    else if (f == kE5M2) return sign | 0x7C;  // infinity
    else return sign | 0x7F;                  // e4m3: NaN
  }
  if (v == 0.0f) return sign;
  int ve;
  std::frexp(v, &ve);
  ve -= 1;  // v = 1.xxx * 2^ve
  unsigned code;
  if (ve < min_exp) {
    code = unsigned(std::lround(v / std::ldexp(1.0f, min_exp - s.mbits)));
  } else {
    const unsigned E = unsigned(ve + s.bias);
    const unsigned M = unsigned(std::lround((v / std::ldexp(1.0f, ve) - 1.0f) * float(1u << s.mbits)));
    code = (E << s.mbits) | M;
  }
  return sign | code;
}


void set_cluster(ClusterState *state) { g_cluster = state; }

int cluster_rank_of(uint64_t cta_key) {
  if (g_cluster == nullptr) return -1;
  for (unsigned r = 0; r < g_cluster->cta_key.size(); ++r)
    if (g_cluster->cta_key[r] == cta_key) return int(r);
  return -1;
}

void mbar_canonical(uint64_t &cta_key, uint32_t &addr) {
  if (g_cluster == nullptr) return;
  unsigned rank;
  uint64_t off;
  dsmem_decode(addr, rank, off);
  if (rank < g_cluster->cta_key.size()) cta_key = g_cluster->cta_key[rank];
  addr = uint32_t(off);
}

void mbar_init(uint64_t cta_key, uint32_t addr, uint32_t count) {
  mbar_canonical(cta_key, addr);
  MbarState &m = g_mbars[{cta_key, addr}];
  m = MbarState();
  m.expected = count;
  m.pending = count;
  mbar_log("init", cta_key, addr, count, &m);
}

void mbar_arrive(uint64_t cta_key, uint32_t addr, uint32_t count) {
  mbar_canonical(cta_key, addr);
  auto it = g_mbars.find({cta_key, addr});
  if (it == g_mbars.end()) {
    mbar_log("arrive", cta_key, addr, count, nullptr);
    return;
  }
  it->second.pending -= count;
  mbar_try_advance(it->second);
  mbar_log("arrive", cta_key, addr, count, &it->second);
}

void mbar_expect_tx(uint64_t cta_key, uint32_t addr, int64_t bytes) {
  mbar_canonical(cta_key, addr);
  auto it = g_mbars.find({cta_key, addr});
  if (it == g_mbars.end()) {
    mbar_log("expect_tx", cta_key, addr, bytes, nullptr);
    return;
  }
  it->second.tx += bytes;
  mbar_log("expect_tx", cta_key, addr, bytes, &it->second);
}

void mbar_complete_tx(uint64_t cta_key, uint32_t addr, int64_t bytes) {
  mbar_canonical(cta_key, addr);
  auto it = g_mbars.find({cta_key, addr});
  if (it == g_mbars.end()) {
    mbar_log("complete_tx", cta_key, addr, bytes, nullptr);
    return;
  }
  it->second.tx -= bytes;
  mbar_try_advance(it->second);
  mbar_log("complete_tx", cta_key, addr, bytes, &it->second);
}

bool mbar_phase_done(uint64_t cta_key, uint32_t addr, uint32_t parity) {
  mbar_canonical(cta_key, addr);
  auto it = g_mbars.find({cta_key, addr});
  if (it == g_mbars.end()) {
    static bool warned = false;
    if (!warned) {
      std::printf("yasai-func: mbarrier wait on 0x%x without init; treated as complete\n", addr);
      warned = true;
    }
    return true;
  }
  const bool done = (it->second.phase & 1u) != (parity & 1u);
  if (mbar_debug()) mbar_log(done ? "wait_true" : "wait_false", cta_key, addr, parity, &it->second);
  return done;
}

void mbar_inval(uint64_t cta_key, uint32_t addr) { mbar_canonical(cta_key, addr); g_mbars.erase({cta_key, addr}); }

namespace {

// Registered function symbols in handle order, and the reverse map.
std::vector<const symbol *> g_func_symbols;
std::unordered_map<const symbol *, uint64_t> g_func_handles;

}  // namespace

uint64_t function_handle(const symbol *s) {
  auto it = g_func_handles.find(s);
  if (it != g_func_handles.end()) return it->second;
  const uint64_t h = kFunctionHandleTag | uint64_t(g_func_symbols.size());
  g_func_symbols.push_back(s);
  g_func_handles.emplace(s, h);
  return h;
}

const symbol *function_of(uint64_t value) {
  if ((value & 0xffff000000000000ull) != kFunctionHandleTag) return nullptr;
  const uint64_t idx = value & 0x0000ffffffffffffull;
  return idx < g_func_symbols.size() ? g_func_symbols[idx] : nullptr;
}

}  // namespace yasai_func
