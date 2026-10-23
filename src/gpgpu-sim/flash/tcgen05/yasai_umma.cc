// yasai: functional model of the Blackwell tensor core instructions
// (tcgen05.mma, .ld, .st, .cp, .commit) for the pure functional mode.
//
// Covered:
// * tcgen05.mma (and .ws) with .kind::f16, ::tf32, ::f8f6f4, ::i8, ::mxf8f6f4,
// ::mxf4 and
//   ::mxf4nvf4 (block scaled with ue8m0 or ue4m3 scale factors in TMEM), A from
//   shared memory or TMEM, B from shared memory, f32 or s32 accumulators,
//   .cta_group::1 and ::2 (a CTA pair of a cluster: A split by rows, B split by
//   columns, each CTA's TMEM receives its rows).
// * Shared memory matrix descriptors in every canonical layout of the PTX ISA:
// K-major and
//   MN-major, no swizzle, 32/64/128-byte swizzles and the 128-byte swizzle with
//   32-byte atoms. Swizzles act on absolute shared addresses, as the hardware
//   does.
// * TMEM accumulator layouts of CUTLASS (cute/atom/mma_traits_sm100.hpp,
// tmem_frg): 1-SM M=64
//   (16-lane quarters) and M=128, 2-SM M=64/128/256 per pair, and the .ws
//   layouts.
// * tcgen05.ld/st shapes 32x32b, 16x64b, 16x128b, 16x256b and 16x32bx2 with .xN
// repetitions and
//   .pack::16b/.unpack::16b, following the thread-value maps of CUTLASS's
//   SM100_TMEM_LOAD traits.
// * tcgen05.cp from a canonical K-major shared layout into TMEM, with
// .warpx2/.warpx4 replication.
// * tcgen05.commit: immediate arrive on the mbarrier (multicast to the CTAs of
// ctaMask). Not covered: sparse (.sp) MMA.
#include "../../../cuda-sim/ptx_ir.h"
#include "../../../cuda-sim/ptx_sim.h"
#include "../../../cuda-sim/yasai_func.h"
#include "../../gpu-sim.h"
#include "tmem.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <vector>

class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"

namespace flash_gpgpu_sim {
namespace yumma {

namespace {

// True when the instruction carries option opt.
bool has(const ptx_instruction *pI, int opt) {
  for (int o : pI->get_options())
    if (o == opt)
      return true;
  return false;
}

// Decoded shared memory matrix descriptor (SM100 format).
struct SmemDesc {
  uint32_t start;  // start address in bytes (18 bits)
  uint32_t lbo;    // leading dimension byte offset
  uint32_t sbo;    // stride dimension byte offset
  unsigned layout; // 0 none, 1 128B with 32B atoms, 2 128B, 4 64B, 6 32B
};

SmemDesc decode_desc(uint64_t d) {
  SmemDesc s;
  s.start = uint32_t((d & 0x3FFF) << 4);
  s.lbo = uint32_t(((d >> 16) & 0x3FFF) << 4);
  s.sbo = uint32_t(((d >> 32) & 0x3FFF) << 4);
  s.layout = unsigned((d >> 61) & 0x7);
  return s;
}

// Row width in bytes of a swizzled layout (0 without swizzling).
unsigned swizzle_width(unsigned layout) {
  switch (layout) {
  case 1:
  case 2:
    return 128;
  case 4:
    return 64;
  case 6:
    return 32;
  default:
    return 0;
  }
}

// Applies the layout's swizzle to an absolute shared address.
uint32_t swizzle(uint32_t a, unsigned layout) {
  switch (layout) {
  case 2:
    return a ^ (((a >> 7) & 7u) << 4); // Swizzle<3,4,3>
  case 4:
    return a ^ (((a >> 7) & 3u) << 4); // Swizzle<2,4,3>
  case 6:
    return a ^ (((a >> 7) & 1u) << 4); // Swizzle<1,4,3>
  case 1:
    return a ^ (((a >> 7) & 3u) << 5); // Swizzle<2,5,2>, 32-byte atoms
  default:
    return a;
  }
}

// Address of 16-byte chunk c (0 or 1; one MMA covers 32 bytes of K) of row mn
// of a K-major operand.
uint32_t kmajor_chunk(const SmemDesc &d, unsigned mn, unsigned c) {
  const unsigned w = swizzle_width(d.layout);
  if (w == 0)
    return d.start + (mn / 8) * d.sbo + (mn % 8) * 16 + c * d.lbo;
  return swizzle(d.start + (mn / 8) * d.sbo + (mn % 8) * w + c * 16, d.layout);
}

// Address of byte mnb along M/N at K index k of an MN-major operand.
uint32_t mnmajor_byte(const SmemDesc &d, unsigned mnb, unsigned k) {
  const unsigned w = swizzle_width(d.layout);
  if (w == 0)
    return d.start + (mnb / 16) * d.sbo + (mnb % 16) + (k % 8) * 16 +
           (k / 8) * d.lbo;
  return swizzle(d.start + (mnb / w) * d.lbo + (mnb % w) + (k % 8) * w +
                     (k / 8) * d.sbo,
                 d.layout);
}

// Element formats of the operands.
enum Fmt {
  kF16,
  kBF16,
  kTF32,
  kE4M3,
  kE5M2,
  kE2M3,
  kE3M2,
  kE2M1,
  kU8,
  kS8,
  kE2M1Packed
};

// Bits of one element in shared memory, and elements per 16-byte chunk.
unsigned fmt_bits(Fmt f) {
  switch (f) {
  case kF16:
  case kBF16:
    return 16;
  case kTF32:
    return 32;
  case kE2M3:
  case kE3M2:
    return 6;
  case kE2M1:
  case kE2M1Packed:
    return 4;
  default:
    return 8;
  }
}
unsigned fmt_per_chunk(Fmt f) {
  switch (f) {
  case kF16:
  case kBF16:
    return 8;
  case kTF32:
    return 4;
  case kE2M1Packed:
    return 32;
  default:
    return 16; // 8-bit types, and FP6/FP4 unpacked to 16 per chunk
  }
}

// Value of the raw bits of an element.
double fmt_value(Fmt f, uint32_t v) {
  switch (f) {
  case kF16: {
    uint16_t h = uint16_t(v);
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    double x = e == 0 ? std::ldexp(double(m), -24)
                      : (e == 31 ? (m ? NAN : INFINITY)
                                 : std::ldexp(1.0 + m / 1024.0, int(e) - 15));
    return s ? -x : x;
  }
  case kBF16: {
    uint32_t u = (v & 0xFFFF) << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
  }
  case kTF32: {
    uint32_t u = v & 0xFFFFE000u;
    float x;
    memcpy(&x, &u, 4);
    return x;
  }
  case kE4M3:
    return yasai_func::minifloat_decode(yasai_func::kE4M3, v);
  case kE5M2:
    return yasai_func::minifloat_decode(yasai_func::kE5M2, v);
  case kE2M3:
    return yasai_func::minifloat_decode(yasai_func::kE2M3, v);
  case kE3M2:
    return yasai_func::minifloat_decode(yasai_func::kE3M2, v);
  case kE2M1:
  case kE2M1Packed:
    return yasai_func::minifloat_decode(yasai_func::kE2M1, v);
  case kU8:
    return double(v & 0xFF);
  case kS8:
    return double(int8_t(v & 0xFF));
  }
  return 0;
}

// Reads element (mn, k) of an operand in shared memory (rank bits of the owning
// CTA in rank_bits).
double smem_element(ptx_thread_info *t, const SmemDesc &d, bool mn_major, Fmt f,
                    unsigned mn, unsigned k, uint64_t rank_bits) {
  const unsigned bits = fmt_bits(f);
  if (mn_major) { // byte-aligned types only
    const unsigned bytes = bits / 8;
    uint32_t raw = 0;
    t->m_shared_mem->read(mnmajor_byte(d, mn * bytes, k) | rank_bits, bytes,
                          &raw);
    return fmt_value(f, raw);
  }
  const unsigned per = fmt_per_chunk(f);
  const uint32_t chunk = kmajor_chunk(d, mn, k / per) | uint32_t(rank_bits);
  const unsigned bit = (k % per) * bits;
  uint64_t raw = 0;
  t->m_shared_mem->read(chunk + bit / 8, bits > 8 ? bits / 8 : 2, &raw);
  raw >>= bit % 8;
  return fmt_value(f, uint32_t(raw & ((1ull << bits) - 1)));
}

// TMEM scope of the CTA in hardware CTA slot cta (cta_group is not part of the
// scope: an allocation made with cta_group::2 and a load with cta_group::1
// reach the same memory).
tcgen05_tmem_scope_t scope_of(ptx_thread_info *t, unsigned cta) {
  unsigned sm = t->get_hw_sid();
  if (sm == unsigned(-1))
    sm = 0;
  return tcgen05_tmem_scope_t{sm, cta, 1};
}

unsigned own_cta(ptx_thread_info *t) {
  unsigned c = t->get_hw_ctaid();
  return c == unsigned(-1) ? unsigned(t->get_flat_ctaid()) : c;
}

// (lane, column) of accumulator element (m, n) of one CTA (m local to the CTA).
void d_cell(unsigned m_cta, unsigned n_total, bool pair, bool ws, unsigned m,
            unsigned n, unsigned &lane, unsigned &col) {
  if (!pair && !ws) {
    if (m_cta == 64) {
      lane = (m % 16) + 32 * (m / 16);
      col = n;
      return;
    }
    lane = m;
    col = n;
    return;
  }
  // 2-SM and weight-stationary layouts: the M_cta x N tile fills 128 lanes by
  // splitting N.
  const unsigned parts = 128 / m_cta; // 1, 2 or 4
  const unsigned span = n_total / parts;
  lane = m + m_cta * (n / span);
  col = n % span;
}

// Scale factor value for row/column x and K block j from TMEM (sf_id: data ID,
// the byte offset of the first scale factor in each TMEM word, from the
// instruction descriptor).
double scale_factor(tcgen05_tmem_manager_t &mgr, const tcgen05_tmem_scope_t &sc,
                    uint32_t addr, unsigned sf_id, unsigned x, unsigned j,
                    bool ue8m0) {
  const unsigned byte =
      sf_id + j; // the data ID is a byte offset in the TMEM word
  const unsigned lane = ((addr >> 16) & 0x7Fu) + (x % 32);
  const unsigned col = (addr & 0xFFFFu) + x / 32 + byte / 4;
  const uint32_t word = mgr.peek(sc, lane, col);
  const uint32_t b = (word >> (8 * (byte % 4))) & 0xFF;
  if (ue8m0)
    return yasai_func::minifloat_decode(yasai_func::kUE8M0, b);
  return yasai_func::minifloat_decode(yasai_func::kE4M3, b & 0x7F);
}

} // namespace

// tcgen05.mma{.ws}.cta_group::{1,2}.kind::* [d], a, b_desc, idesc{, [sfa],
// [sfb]}{, mask}, en_d
void mma(const ptx_instruction *pI, ptx_thread_info *t) {
  if (t->get_laneid() != 0)
    return;
  const bool pair = has(pI, TCGEN05_CTA_GROUP_2_OPTION);
  const bool ws = has(pI, TCGEN05_WS_OPTION);
  const bool block_scale = has(pI, TCGEN05_BLOCK_SCALE_OPTION);
  enum { F16, TF32, F8F6F4, I8, MXF8F6F4, MXF4, MXF4NVF4 } kind = F16;
  if (has(pI, TCGEN05_KIND_TF32_OPTION))
    kind = TF32;
  else if (has(pI, TCGEN05_KIND_F8F6F4_OPTION))
    kind = F8F6F4;
  else if (has(pI, TCGEN05_KIND_I8_OPTION))
    kind = I8;
  else if (has(pI, TCGEN05_KIND_MXF8F6F4_OPTION))
    kind = MXF8F6F4;
  else if (has(pI, TCGEN05_KIND_MXF4_OPTION))
    kind = MXF4;
  else if (has(pI, TCGEN05_KIND_MXF4NVF4_OPTION))
    kind = MXF4NVF4;

  const unsigned nops = pI->get_num_operands();
  auto u32 = [&](unsigned i) {
    const operand_info &op = pI->operand_lookup(i);
    return t->get_operand_value(op, op, B32_TYPE, t, 0).u32;
  };
  auto u64 = [&](unsigned i) {
    const operand_info &op = pI->operand_lookup(i);
    return t->get_operand_value(op, op, B64_TYPE, t, 1).u64;
  };
  const uint32_t d_addr = u32(0);
  const bool a_tmem = pI->operand_lookup(1).is_memory_operand();
  const uint32_t a_tmem_addr = a_tmem ? u32(1) : 0;
  const uint64_t a_desc_v = a_tmem ? 0 : u64(1);
  const uint64_t b_desc_v = u64(2);
  const uint32_t idesc = u32(3);
  uint32_t sfa_addr = 0, sfb_addr = 0;
  unsigned next = 4;
  if (block_scale) {
    sfa_addr = u32(4);
    sfb_addr = u32(5);
    next = 6;
  }
  while (next + 1 < nops && pI->operand_lookup(next).is_vector())
    next++; // disable-output mask
  // enable-input-d follows the operands above. After it come scale-input-d
  // (non-.ws, an immediate: D = A * B + D * 2^-scale) or the zero-column mask
  // descriptor (.ws), so it is not always the last operand.
  bool enable_d = true;
  double d_scale = 1.0;
  if (next < nops) {
    const operand_info &op = pI->operand_lookup(next);
    if (op.is_reg() && op.get_symbol() &&
        op.get_symbol()->type()->get_key().scalar_type() == PRED_TYPE)
      enable_d = (t->get_operand_value(op, op, PRED_TYPE, t, 1).pred & 1) == 0;
    else
      enable_d = t->get_operand_value(op, op, B32_TYPE, t, 1).u32 != 0;
    if (!ws && next + 1 < nops)
      d_scale = std::ldexp(1.0, -int(u32(next + 1)));
    if (ws && next + 1 < nops && u64(next + 1) != 0) {
      fprintf(
          stderr,
          "yasai-umma: tcgen05.mma.ws zero-column mask is not implemented\n");
      abort();
    }
  }

  // Instruction descriptor (PTX ISA 9.7.18.4.2). The block-scaled kinds (Table
  // 53) use another layout than kind::f16/tf32/f8f6f4/i8 (Table 51): bits 4-5
  // hold the SFB data ID instead of the D type (D is f32), bits 29-30 the SFA
  // data ID, bits 27-28 M >> 7, bit 23 (mxf8f6f4) or bits 23-24 (mxf4/mxf4nvf4)
  // the scale type, and bit 31 (with bit 3 for mxf4) K.
  const bool bs_kind = kind == MXF8F6F4 || kind == MXF4 || kind == MXF4NVF4;
  const unsigned c_format = bs_kind ? 1u : (idesc >> 4) & 3;
  const unsigned a_format = (idesc >> 7) & 7, b_format = (idesc >> 10) & 7;
  const bool a_neg = (idesc >> 13) & 1, b_neg = (idesc >> 14) & 1;
  const bool a_mn = (idesc >> 15) & 1, b_mn = (idesc >> 16) & 1;
  const unsigned N = ((idesc >> 17) & 0x3F) << 3;
  const unsigned M =
      bs_kind ? ((idesc >> 27) & 3) << 7 : ((idesc >> 24) & 0x1F) << 4;
  const unsigned scale_type =
      kind == MXF8F6F4 ? (idesc >> 23) & 1 : (idesc >> 23) & 3;
  const bool scale_ue8m0 = scale_type == 1;
  if (bs_kind && scale_type > 1) {
    fprintf(stderr, "yasai-umma: scale type %u (ue5m3) is not implemented\n",
            scale_type);
    abort();
  }
  const unsigned sfb_id = bs_kind ? (idesc >> 4) & 3 : 0;
  const unsigned sfa_id = bs_kind ? (idesc >> 29) & 3 : 0;
  if ((idesc >> 2) & 1) {
    fprintf(stderr, "yasai-umma: sparse tcgen05.mma is not implemented\n");
    abort();
  }
  auto fmt_of = [&](unsigned code) -> Fmt {
    switch (kind) {
    case F16:
      return code == 1 ? kBF16 : kF16;
    case TF32:
      return kTF32;
    case I8:
      return code == 1 ? kS8 : kU8;
    case MXF4:
    case MXF4NVF4:
      return kE2M1Packed;
    default:
      switch (code) {
      case 0:
        return kE4M3;
      case 1:
        return kE5M2;
      case 3:
        return kE2M3;
      case 4:
        return kE3M2;
      default:
        return kE2M1;
      }
    }
  };
  const Fmt fa = fmt_of(a_format), fb = fmt_of(b_format);
  unsigned K = 16;
  switch (kind) {
  case F16:
    K = 16;
    break;
  case TF32:
    K = 8;
    break;
  case MXF4:
  case MXF4NVF4:
    K = 64 +
        32 * ((((idesc >> 3) & 1) << 1) | ((idesc >> 31) & 1)); // 64, 96 or 128
    break;
  case MXF8F6F4:
    K = (idesc >> 31) & 1 ? 64 : 32;
    break;
  default:
    K = 32;
    break;
  }
  unsigned nsf = 1; // scale factors per row per MMA
  if (block_scale) {
    if (has(pI, TCGEN05_SCALE_VEC_4X_OPTION) || has(pI, TCGEN05_BLOCK16_OPTION))
      nsf = K / 16;
    else if (has(pI, TCGEN05_SCALE_VEC_2X_OPTION) ||
             has(pI, TCGEN05_BLOCK32_OPTION))
      nsf = K / 32;
    else if (has(pI, TCGEN05_SCALE_VEC_1X_OPTION))
      nsf = 1;
    else
      nsf = (kind == MXF4NVF4) ? K / 16 : K / 32;
    if (nsf == 0)
      nsf = 1;
  }
  const unsigned block = K / nsf;

  tcgen05_tmem_manager_t &mgr = t->get_gpu()->get_tcgen05_tmem_manager();
  const unsigned me = own_cta(t);
  const unsigned nctas = pair ? 2 : 1;
  const unsigned m_cta = M / nctas;
  const unsigned n_cta = N / nctas; // B columns held by each CTA
  yasai_func::ClusterState *cl = yasai_func::cluster();
  unsigned my_rank = 0;
  if (cl != nullptr)
    my_rank = yasai_func::current_rank();
  const unsigned lead_rank = pair ? (my_rank & ~1u) : my_rank;
  const unsigned lead_cta = pair ? (me & ~1u) : me;
  if (pair && cl == nullptr) {
    fprintf(stderr, "yasai-umma: tcgen05.mma.cta_group::2 outside a cluster\n");
    abort();
  }
  const SmemDesc ad = decode_desc(a_desc_v), bd = decode_desc(b_desc_v);

  // B (N x K): columns [c * n_cta, (c + 1) * n_cta) live in CTA c of the pair.
  std::vector<double> B(size_t(N) * K);
  for (unsigned n = 0; n < N; ++n) {
    const unsigned c = n / n_cta;
    const uint64_t bits = cl != nullptr ? (uint64_t(lead_rank + c) << 24) : 0;
    for (unsigned k = 0; k < K; ++k)
      B[size_t(n) * K + k] =
          smem_element(t, bd, b_mn, fb, n % n_cta, k, bits) * (b_neg ? -1 : 1);
  }
  for (unsigned c = 0; c < nctas; ++c) {
    const tcgen05_tmem_scope_t sc = scope_of(t, lead_cta + c);
    const uint64_t bits = cl != nullptr ? (uint64_t(lead_rank + c) << 24) : 0;
    const uint32_t d_lane0 = (d_addr >> 16) & 0xFFFF, d_col0 = d_addr & 0xFFFF;
    for (unsigned m = 0; m < m_cta; ++m) {
      std::vector<double> a(K);
      for (unsigned k = 0; k < K; ++k) {
        double v;
        if (a_tmem) {
          // A in TMEM (PTX ISA 9.7.18.10.4): 16/32-bit types packed in 32-bit
          // words; FP8 and the FP6/FP4 types of .kind::f8f6f4/mxf8f6f4 one per
          // byte (E2M1 in bits [5:2], FP6 in bits [5:0]); .kind::mxf4/mxf4nvf4
          // two FP4 per byte.
          unsigned slot_bits = fmt_bits(fa);
          if (fa == kE2M1 || fa == kE2M3 || fa == kE3M2)
            slot_bits = 8;
          const unsigned per_word = 32 / slot_bits;
          unsigned lane, col;
          d_cell(m_cta, K / per_word, pair, ws, m, k / per_word, lane, col);
          const uint32_t w = mgr.peek(sc, ((a_tmem_addr >> 16) & 0xFFFF) + lane,
                                      (a_tmem_addr & 0xFFFF) + col);
          uint32_t slot =
              (w >> (slot_bits * (k % per_word))) & ((1ull << slot_bits) - 1);
          if (fa == kE2M1)
            slot = (slot >> 2) & 0xF;
          else if (fa == kE2M3 || fa == kE3M2)
            slot &= 0x3F;
          v = fmt_value(fa, slot);
        } else {
          v = smem_element(t, ad, a_mn, fa, m, k, bits);
        }
        a[k] = v * (a_neg ? -1 : 1);
      }
      std::vector<double> sa(nsf, 1.0);
      if (block_scale)
        for (unsigned j = 0; j < nsf; ++j)
          sa[j] = scale_factor(mgr, sc, sfa_addr, sfa_id, m, j, scale_ue8m0);
      for (unsigned n = 0; n < N; ++n) {
        unsigned lane, col;
        d_cell(m_cta, N, pair, ws, m, n, lane, col);
        const uint32_t old =
            enable_d ? mgr.peek(sc, d_lane0 + lane, d_col0 + col) : 0;
        if (kind == I8) {
          int64_t acc = enable_d ? int64_t(int32_t(old)) : 0;
          for (unsigned k = 0; k < K; ++k)
            acc += int64_t(a[k]) * int64_t(B[size_t(n) * K + k]);
          if ((idesc >> 3) & 1)
            acc = acc > INT32_MAX ? INT32_MAX
                                  : (acc < INT32_MIN ? INT32_MIN : acc);
          mgr.poke(sc, d_lane0 + lane, d_col0 + col, uint32_t(int32_t(acc)));
          continue;
        }
        double acc = 0;
        for (unsigned j = 0; j < nsf; ++j) {
          double part = 0;
          for (unsigned k = j * block; k < (j + 1) * block; ++k)
            part += a[k] * B[size_t(n) * K + k];
          if (block_scale)
            part *= sa[j] *
                    scale_factor(mgr, sc, sfb_addr, sfb_id, n, j, scale_ue8m0);
          acc += part;
        }
        if (c_format ==
            0) { // f16 accumulator, one per 32-bit word in the low 16 bits
          const double prev = enable_d ? fmt_value(kF16, old & 0xFFFF) : 0.0;
          const float r = float(prev * d_scale + acc);
          mgr.poke(sc, d_lane0 + lane, d_col0 + col,
                   yasai_func::minifloat_encode_f16(r));
          continue;
        }
        float prev = 0;
        if (enable_d)
          memcpy(&prev, &old, 4);
        const float r = float(double(prev) * d_scale + acc);
        uint32_t ru;
        memcpy(&ru, &r, 4);
        mgr.poke(sc, d_lane0 + lane, d_col0 + col, ru);
      }
    }
  }
}

// Thread-value map of one repetition of a tcgen05.ld/st shape: register r of
// thread lane maps to (dp, column) relative to the address; cols is the columns
// one repetition covers.
static void ldst_cell(unsigned shape, unsigned lane, unsigned r,
                      unsigned imm_half, unsigned &dp, unsigned &col) {
  switch (shape) {
  case 0: // 32x32b: thread t, lane t, one column
    dp = lane;
    col = 0;
    return;
  case 1: // 16x64b: dp (t/4) + 8*(t%2), column (t/2)%2
    dp = (lane / 4) + 8 * (lane % 2);
    col = (lane / 2) % 2;
    return;
  case 2: // 16x128b: dp t/4 (+8 for register 1), column t%4
    dp = lane / 4 + 8 * r;
    col = lane % 4;
    return;
  case 3: // 16x256b: dp t/4 (+8 for registers 2,3), columns 2*(t%4) + r%2
    dp = lane / 4 + 8 * (r / 2);
    col = 2 * (lane % 4) + (r % 2);
    return;
  default: // 16x32bx2: dp t%16, second half at the immediate column offset
    dp = lane % 16;
    col = (lane / 16) ? imm_half : 0;
    return;
  }
}

static void ldst_shape(const ptx_instruction *pI, unsigned &shape,
                       unsigned &regs_per_rep, unsigned &cols_per_rep) {
  if (has(pI, TCGEN05_16X64B_OPTION)) {
    shape = 1;
    regs_per_rep = 1;
    cols_per_rep = 2;
  } else if (has(pI, TCGEN05_16X128B_OPTION)) {
    shape = 2;
    regs_per_rep = 2;
    cols_per_rep = 4;
  } else if (has(pI, TCGEN05_16X256B_OPTION)) {
    shape = 3;
    regs_per_rep = 4;
    cols_per_rep = 8;
  } else if (has(pI, TCGEN05_16X32BX2_OPTION)) {
    shape = 4;
    regs_per_rep = 1;
    cols_per_rep = 1;
  } else {
    shape = 0;
    regs_per_rep = 1;
    cols_per_rep = 1;
  }
}

// tcgen05.ld.sync.aligned.<shape>.xN{.pack::16b}.b32 {regs}, [taddr]{, imm}
void ld(const ptx_instruction *pI, ptx_thread_info *t) {
  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &addr_op = pI->operand_lookup(1);
  const uint32_t addr =
      t->get_operand_value(addr_op, addr_op, B32_TYPE, t, 0).u32;
  unsigned imm_half = 0;
  if (pI->get_num_operands() > 2) {
    const operand_info &op = pI->operand_lookup(2);
    imm_half = t->get_operand_value(op, op, B32_TYPE, t, 1).u32;
  }
  unsigned shape, rpr, cpr;
  ldst_shape(pI, shape, rpr, cpr);
  const bool pack = has(pI, TCGEN05_PACK_16B_OPTION);
  const unsigned nregs = dst.get_vect_nelem();
  const unsigned nvals = pack ? nregs * 2 : nregs;
  tcgen05_tmem_manager_t &mgr = t->get_gpu()->get_tcgen05_tmem_manager();
  const tcgen05_tmem_scope_t sc = scope_of(t, own_cta(t));
  const unsigned lane0 = (addr >> 16) & 0xFFFF, col0 = addr & 0xFFFF,
                 lane = t->get_laneid();
  std::vector<uint32_t> vals(nvals);
  for (unsigned v = 0; v < nvals; ++v) {
    unsigned dp, col;
    ldst_cell(shape, lane, v % rpr, imm_half, dp, col);
    vals[v] = mgr.peek(sc, lane0 + dp, col0 + (v / rpr) * cpr + col);
  }
  for (unsigned i = 0; i < nregs; ++i) {
    const symbol *s = dst.vec_symbol(i);
    if (s == NULL || strcmp(s->name().c_str(), "_") == 0)
      continue;
    ptx_reg_t r;
    r.u64 = 0;
    r.u32 = pack ? ((vals[2 * i] & 0xFFFF) | (vals[2 * i + 1] << 16)) : vals[i];
    t->set_reg(s, r);
  }
}

// tcgen05.st.sync.aligned.<shape>.xN{.unpack::16b}.b32 [taddr]{, imm}, {regs}
void st(const ptx_instruction *pI, ptx_thread_info *t) {
  const operand_info &addr_op = pI->operand_lookup(0);
  const operand_info &src = pI->operand_lookup(pI->get_num_operands() - 1);
  const uint32_t addr =
      t->get_operand_value(addr_op, addr_op, B32_TYPE, t, 0).u32;
  unsigned imm_half = 0;
  if (pI->get_num_operands() > 2) {
    const operand_info &op = pI->operand_lookup(1);
    imm_half = t->get_operand_value(op, op, B32_TYPE, t, 1).u32;
  }
  unsigned shape, rpr, cpr;
  ldst_shape(pI, shape, rpr, cpr);
  const bool unpack = has(pI, TCGEN05_UNPACK_16B_OPTION);
  const unsigned nregs = src.get_vect_nelem();
  std::vector<uint32_t> regs(nregs, 0), vals;
  for (unsigned i = 0; i < nregs; ++i) {
    const symbol *s = src.vec_symbol(i);
    if (s != NULL && strcmp(s->name().c_str(), "_") != 0)
      regs[i] = t->get_reg(s).u32;
  }
  for (unsigned i = 0; i < nregs; ++i) {
    if (unpack) {
      vals.push_back(regs[i] & 0xFFFF);
      vals.push_back(regs[i] >> 16);
    } else {
      vals.push_back(regs[i]);
    }
  }
  tcgen05_tmem_manager_t &mgr = t->get_gpu()->get_tcgen05_tmem_manager();
  const tcgen05_tmem_scope_t sc = scope_of(t, own_cta(t));
  const unsigned lane0 = (addr >> 16) & 0xFFFF, col0 = addr & 0xFFFF,
                 lane = t->get_laneid();
  for (unsigned v = 0; v < vals.size(); ++v) {
    unsigned dp, col;
    ldst_cell(shape, lane, v % rpr, imm_half, dp, col);
    mgr.poke(sc, lane0 + dp, col0 + (v / rpr) * cpr + col, vals[v]);
  }
}

// tcgen05.cp.cta_group::{1,2}.<shape>{.warpx2::02_13,.warpx2::01_23,.warpx4}
// [taddr], s_desc: rows x row_bytes from a K-major canonical shared layout into
// TMEM (lane = row, 32-bit columns), replicated to the warp quarters of the
// multicast option; with cta_group::2 both CTAs of the pair copy their own
// shared memory into their own TMEM.
void cp(const ptx_instruction *pI, ptx_thread_info *t) {
  if (t->get_laneid() != 0)
    return;
  unsigned rows = 128, row_bytes = 32;
  if (has(pI, TCGEN05_128X256B_OPTION)) {
    rows = 128;
    row_bytes = 32;
  } else if (has(pI, TCGEN05_128X128B_OPTION)) {
    rows = 128;
    row_bytes = 16;
  } else if (has(pI, TCGEN05_64X128B_OPTION)) {
    rows = 64;
    row_bytes = 16;
  } else if (has(pI, TCGEN05_32X128B_OPTION)) {
    rows = 32;
    row_bytes = 16;
  } else if (has(pI, TCGEN05_4X256B_OPTION)) {
    rows = 4;
    row_bytes = 32;
  }
  std::vector<unsigned> lane_offsets = {0};
  if (has(pI, TCGEN05_WARPX4_OPTION))
    lane_offsets = {0, 32, 64, 96};
  else if (has(pI, TCGEN05_WARPX2_02_13_OPTION))
    lane_offsets = {0, 64};
  else if (has(pI, TCGEN05_WARPX2_01_23_OPTION))
    lane_offsets = {0, 32};
  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src = pI->operand_lookup(1);
  const uint32_t taddr = t->get_operand_value(dst, dst, B32_TYPE, t, 0).u32;
  const SmemDesc d =
      decode_desc(t->get_operand_value(src, src, B64_TYPE, t, 1).u64);
  const bool pair = has(pI, TCGEN05_CTA_GROUP_2_OPTION);
  yasai_func::ClusterState *cl = yasai_func::cluster();
  const unsigned me = own_cta(t);
  const unsigned my_rank = cl != nullptr ? yasai_func::current_rank() : 0;
  tcgen05_tmem_manager_t &mgr = t->get_gpu()->get_tcgen05_tmem_manager();
  const unsigned lane0 = (taddr >> 16) & 0xFFFF, col0 = taddr & 0xFFFF;
  for (unsigned c = 0; c < (pair ? 2u : 1u); ++c) {
    const unsigned cta = pair ? ((me & ~1u) + c) : me;
    const uint64_t bits =
        cl != nullptr ? (uint64_t(pair ? ((my_rank & ~1u) + c) : my_rank) << 24)
                      : 0;
    const tcgen05_tmem_scope_t sc = scope_of(t, cta);
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned ch = 0; ch < row_bytes / 16; ++ch) {
        uint32_t w[4];
        t->m_shared_mem->read(kmajor_chunk(d, r, ch) | uint32_t(bits), 16, w);
        for (unsigned off : lane_offsets)
          for (unsigned i = 0; i < 4; ++i)
            mgr.poke(sc, lane0 + off + r, col0 + ch * 4 + i, w[i]);
      }
  }
}

// tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one{.shared::cluster}{.multicast::cluster}
// .b64 [mbar]{, ctaMask}: the asynchronous tcgen05 operations of the thread are
// already complete in the functional mode, so the mbarrier gets its arrival now
// (in every CTA of ctaMask with .multicast::cluster).
void commit(const ptx_instruction *pI, ptx_thread_info *t) {
  if (t->get_laneid() != 0)
    return;
  const operand_info &bar = pI->operand_lookup(0);
  const uint32_t addr = t->get_operand_value(bar, bar, B32_TYPE, t, 0).u32;
  if (has(pI, TCGEN05_MULTICAST_CLUSTER_OPTION) && pI->get_num_operands() > 1 &&
      yasai_func::cluster() != nullptr) {
    const operand_info &m = pI->operand_lookup(1);
    const uint32_t mask =
        t->get_operand_value(m, m, B32_TYPE, t, 1).u32 & 0xFFFF;
    unsigned rank;
    uint64_t off;
    yasai_func::dsmem_decode(addr, rank, off);
    for (unsigned r = 0; r < yasai_func::cluster()->size; ++r)
      if ((mask >> r) & 1)
        yasai_func::mbar_arrive(t->get_cta_uid(),
                                uint32_t(yasai_func::dsmem_encode(r, off)), 1);
    return;
  }
  yasai_func::mbar_arrive(t->get_cta_uid(), addr, 1);
}

} // namespace yumma
} // namespace flash_gpgpu_sim
