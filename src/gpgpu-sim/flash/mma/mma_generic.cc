// yasai: generic functional model of mma.sync for the element types the
// type-specific files do not cover: FP8 (e4m3/e5m2), FP6/FP4 of .kind::f8f6f4
// (8-bit containers), the block-scaled kinds
// .kind::mxf8f6f4/.kind::mxf4/.kind::mxf4nvf4 (ue8m0 or ue4m3 scales), s8/u8,
// s4/u4, b1 (.and.popc/.xor.popc) and f64, with f32, f16, s32 or f64
// accumulators.
//
// Fragment layouts (PTX ISA 9.7.14.5): A is row-major M x K and B column-major
// K x N. A 32-bit register (64-bit for f64) holds p = width / bits consecutive
// K elements, K is cut into chunks of 4p, and the A registers of a lane walk
// rows groupID and groupID + 8 within a chunk before the next chunk; B
// registers walk the chunks. C/D hold one element per register (two f16 per
// register for f16 accumulators): row groupID (+8 for the second pair), column
// threadID_in_group * 2 + (i & 1). Block scaling (PTX ISA 9.7.14.4): SF_A of
// row m, K chunk s is byte byte-id-a + s of scale-a-data in lane (m % 8) * 4 +
// thread-id-a * 2 + m / 8; SF_B of column n is byte byte-id-b + s of
// scale-b-data in lane n * 4 + thread-id-b. tests/mma_types.cu (fakeenv) checks
// these layouts on an RTX 5090 and in the simulator.

#include "../../../cuda-sim/yasai_func.h"
#include "tensor_mma.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <vector>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "../../../cuda-sim/ptx_sim.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

// Element value of the low code bits of v in PTX type t (code without container
// padding).
double elem_value(int t, uint64_t v) {
  using namespace yasai_func;
  switch (t) {
  case E4M3_TYPE:
    return minifloat_decode(kE4M3, unsigned(v & 0xFF));
  case E5M2_TYPE:
    return minifloat_decode(kE5M2, unsigned(v & 0xFF));
  case E2M1_TYPE:
    return minifloat_decode(kE2M1, unsigned(v & 0xF));
  case E2M3_TYPE:
    return minifloat_decode(kE2M3, unsigned(v & 0x3F));
  case E3M2_TYPE:
    return minifloat_decode(kE3M2, unsigned(v & 0x3F));
  case UE8M0_TYPE:
    return minifloat_decode(kUE8M0, unsigned(v & 0xFF));
  case UE4M3_TYPE:
    return minifloat_decode(kE4M3, unsigned(v & 0x7F));
  case S8_TYPE:
    return double(int8_t(v & 0xFF));
  case U8_TYPE:
    return double(v & 0xFF);
  case S4_TYPE:
    return double(int(v & 7) - int(v & 8));
  case U4_TYPE:
    return double(v & 15);
  case B1_TYPE:
    return double(v & 1);
  case F64_TYPE: {
    double d;
    memcpy(&d, &v, 8);
    return d;
  }
  default:
    fprintf(stderr, "yasai-mma: element type %d is not supported\n", t);
    abort();
  }
}

// Width of the element code of type t in bits.
unsigned code_bits(int t) {
  switch (t) {
  case E2M1_TYPE:
  case S4_TYPE:
  case U4_TYPE:
    return 4;
  case E2M3_TYPE:
  case E3M2_TYPE:
    return 6;
  case B1_TYPE:
    return 1;
  case F64_TYPE:
    return 64;
  default:
    return 8;
  }
}

// Reads the n registers of a vector operand (or of a scalar register operand
// when n is 1).
void read_regs(ptx_thread_info *t, const operand_info &op, ptx_reg_t *out,
               unsigned n) {
  if (op.is_vector()) {
    t->get_vector_operand_values(op, out, n);
  } else {
    out[0] = t->get_reg(op.get_symbol());
  }
}

} // namespace

// Generic mma.sync: decodes all lanes' fragments into full matrices, computes
// D = (A * SF_A) * (B * SF_B) + C in double precision, rounds to the
// accumulator type and writes the D fragments back.
//   pI   the mma instruction; M, N, K its shape; tid the first thread of the
//   warp.
void tensor_mma_generic_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t inst, int M, int N, int K,
                             unsigned tid) {
  (void)inst;
  // Types: dtype, atype, btype, ctype and, with .block_scale, stype.
  const std::list<int> type_list = pI->get_scalar_type();
  std::vector<int> types(type_list.begin(), type_list.end());
  if (types.size() < 4) {
    fprintf(stderr, "yasai-mma: mma with %zu types is not supported\n",
            types.size());
    abort();
  }
  const int dtype = types[0], atype = types[1], btype = types[2];
  const int stype = types.size() > 4 ? types[4] : 0;
  bool packed4 = false, block_scale = false, popc_xor = false;
  int scale_vec = 0;
  for (int opt : pI->get_options()) {
    switch (opt) {
    case TCGEN05_KIND_MXF4_OPTION:
    case TCGEN05_KIND_MXF4NVF4_OPTION:
      packed4 = true;
      break;
    case TCGEN05_BLOCK_SCALE_OPTION:
      block_scale = true;
      break;
    case TCGEN05_SCALE_VEC_1X_OPTION:
      scale_vec = 1;
      break;
    case TCGEN05_SCALE_VEC_2X_OPTION:
      scale_vec = 2;
      break;
    case TCGEN05_SCALE_VEC_4X_OPTION:
      scale_vec = 4;
      break;
    case ATOMIC_XOR:
      popc_xor = true;
      break;
    default:
      break;
    }
  }
  if (block_scale && scale_vec == 0)
    scale_vec = packed4 ? 2 : 1; // PTX defaults
  const int sv = block_scale ? scale_vec : 1;
  const bool sat = pI->get_mma_saturate();

  // Container bits per element in the A/B registers: FP4/FP6 sit in 8-bit
  // containers except under .kind::mxf4/mxf4nvf4; integer and FP8 types are
  // packed at their width.
  auto container = [&](int t) -> unsigned {
    if ((t == E2M1_TYPE && !packed4) || t == E2M3_TYPE || t == E3M2_TYPE)
      return 8;
    return code_bits(t);
  };
  const unsigned abits = container(atype), bbits = container(btype);
  const unsigned rw = (atype == F64_TYPE) ? 64 : 32;
  const unsigned pa = rw / abits, pb = rw / bbits;
  const unsigned na = unsigned(M * K) / (32 * pa),
                 nb = unsigned(K * N) / (32 * pb);
  const bool c_f16 = (dtype == F16_TYPE);
  const unsigned nc = c_f16 ? unsigned(M * N) / 64 : unsigned(M * N) / 32;
  if (na > 8 || nb > 8 || nc > 8) {
    fprintf(stderr,
            "yasai-mma: fragment of %u/%u/%u registers is not supported\n", na,
            nb, nc);
    abort();
  }
  // Value bits of an element inside its container (E2M1 in 8 bits sits in bits
  // [5:2]).
  auto field = [](int t, unsigned cbits, uint64_t container_bits) -> uint64_t {
    if (cbits == 8 && t == E2M1_TYPE)
      return (container_bits >> 2) & 0xF;
    if (cbits == 8 && (t == E2M3_TYPE || t == E3M2_TYPE))
      return container_bits & 0x3F;
    return container_bits;
  };

  std::vector<double> A(M * K), B(K * N), C(M * N), SA(M * sv, 1.0),
      SB(sv * N, 1.0);
  const operand_info &oa = pI->operand_lookup(1), &ob = pI->operand_lookup(2),
                     &oc = pI->operand_lookup(3);
  for (unsigned lane = 0; lane < 32; lane++) {
    ptx_thread_info *t = core->get_thread_info()[tid + lane];
    const unsigned g = lane >> 2, q = lane & 3;
    ptx_reg_t ra[8], rb[8], rc[8];
    read_regs(t, oa, ra, na);
    read_regs(t, ob, rb, nb);
    read_regs(t, oc, rc, nc);
    for (unsigned r = 0; r < na; r++) {
      const unsigned cw = 4 * pa, chunk = r / unsigned(M / 8),
                     m = g + 8 * (r % unsigned(M / 8));
      for (unsigned j = 0; j < pa; j++) {
        const unsigned k = chunk * cw + q * pa + j;
        const uint64_t bits =
            rw == 64 ? ra[r].u64
                     : (uint64_t(ra[r].u32) >> (j * abits)) &
                           ((abits >= 64) ? ~0ull : ((1ull << abits) - 1));
        A[m * K + k] = elem_value(atype, field(atype, abits, bits));
      }
    }
    for (unsigned r = 0; r < nb; r++) {
      const unsigned cw = 4 * pb;
      for (unsigned j = 0; j < pb; j++) {
        const unsigned k = r * cw + q * pb + j;
        const uint64_t bits =
            rw == 64 ? rb[r].u64
                     : (uint64_t(rb[r].u32) >> (j * bbits)) &
                           ((bbits >= 64) ? ~0ull : ((1ull << bbits) - 1));
        B[k * N + g] = elem_value(btype, field(btype, bbits, bits));
      }
    }
    for (unsigned r = 0; r < nc; r++) {
      if (c_f16) {
        const unsigned m = g + 8 * r;
        for (unsigned j = 0; j < 2; j++)
          C[m * N + q * 2 + j] =
              mma_f16_to_f32(uint16_t(rc[r].u32 >> (16 * j)));
      } else {
        const unsigned m = g + 8 * (r / 2), n = q * 2 + (r % 2);
        double v;
        if (dtype == S32_TYPE)
          v = double(rc[r].s32);
        else if (dtype == F64_TYPE)
          v = rc[r].f64;
        else
          v = double(rc[r].f32);
        C[m * N + n] = v;
      }
    }
  }
  if (block_scale) {
    // Operands 4-7: scale-a-data, {byte-id-a, thread-id-a}, scale-b-data,
    // {byte-id-b, thread-id-b}.
    ptx_reg_t sel_a[2], sel_b[2];
    ptx_thread_info *t0 = core->get_thread_info()[tid];
    read_regs(t0, pI->operand_lookup(5), sel_a, 2);
    read_regs(t0, pI->operand_lookup(7), sel_b, 2);
    const unsigned bida = sel_a[0].u16, tida = sel_a[1].u16,
                   bidb = sel_b[0].u16, tidb = sel_b[1].u16;
    for (int m = 0; m < M; m++) {
      ptx_reg_t w;
      read_regs(core->get_thread_info()[tid + (m % 8) * 4 + tida * 2 + m / 8],
                pI->operand_lookup(4), &w, 1);
      for (int s = 0; s < sv; s++)
        SA[m * sv + s] = elem_value(stype, (w.u32 >> (8 * (bida + s))) & 0xFF);
    }
    for (int n = 0; n < N; n++) {
      ptx_reg_t w;
      read_regs(core->get_thread_info()[tid + n * 4 + tidb],
                pI->operand_lookup(6), &w, 1);
      for (int s = 0; s < sv; s++)
        SB[s * N + n] = elem_value(stype, (w.u32 >> (8 * (bidb + s))) & 0xFF);
    }
  }

  const int kblk = K / sv;
  const operand_info &od = pI->operand_lookup(0);
  std::vector<ptx_reg_t> D(M * N);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      ptx_reg_t &d = D[m * N + n];
      d.u64 = 0;
      if (atype == B1_TYPE || dtype == S32_TYPE) {
        int64_t acc = int64_t(C[m * N + n]);
        for (int k = 0; k < K; k++) {
          const int64_t a = int64_t(A[m * K + k]), b = int64_t(B[k * N + n]);
          if (atype == B1_TYPE)
            acc += popc_xor ? (a ^ b) : (a & b);
          else
            acc += a * b;
        }
        if (sat) {
          if (acc > INT32_MAX)
            acc = INT32_MAX;
          if (acc < INT32_MIN)
            acc = INT32_MIN;
        }
        d.s32 = int32_t(uint32_t(uint64_t(acc)));
        continue;
      }
      double acc = C[m * N + n];
      for (int k = 0; k < K; k++)
        acc += A[m * K + k] * B[k * N + n] * SA[m * sv + k / kblk] *
               SB[(k / kblk) * N + n];
      if (dtype == F64_TYPE)
        d.f64 = acc;
      else if (c_f16)
        d.u32 = yasai_func::minifloat_encode_f16(float(acc));
      else
        d.f32 = float(acc);
    }
  }
  for (unsigned lane = 0; lane < 32; lane++) {
    ptx_thread_info *t = core->get_thread_info()[tid + lane];
    const unsigned g = lane >> 2, q = lane & 3;
    ptx_reg_t out[8];
    for (unsigned r = 0; r < nc; r++) {
      if (c_f16) {
        const unsigned m = g + 8 * r;
        out[r].u64 = 0;
        out[r].u32 =
            (D[m * N + q * 2].u32 & 0xFFFF) | (D[m * N + q * 2 + 1].u32 << 16);
      } else {
        out[r] = D[(g + 8 * (r / 2)) * N + q * 2 + (r % 2)];
      }
    }
    if (od.is_vector()) {
      for (unsigned r = 0; r < nc; r++)
        t->set_reg(od.vec_symbol(r), out[r]);
    } else {
      t->set_reg(od.get_symbol(), out[0]);
    }
  }
}

} // namespace flash_gpgpu_sim
