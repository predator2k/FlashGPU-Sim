// yasai: functional model of the sparse warp-level MMA, mma.sp{::ordered_metadata}.sync.aligned
// (PTX ISA 9.7.16.6), with structured-sparse A: f16/bf16 (m16n8k16, m16n8k32), tf32
// (m16n8k8, m16n8k16), s8/u8 (m16n8k32, m16n8k64), FP8/FP6/FP4 of .kind::f8f6f4 (m16n8k64),
// s4/u4 (m16n8k64, m16n8k128) and e2m1 of .kind::mxf4/.kind::mxf4nvf4 without scaling
// (m16n8k128), with f32, f16 or s32 accumulators.
//
// Storage (PTX ISA 9.7.16.6.1): A is M x K with half of every row zero; its operand holds the
// M x K/2 packed non-zero elements in the fragment layout of a dense A of shape M x K/2. The
// metadata operand e gives the positions. Positions are counted in "units": an element for
// 8-bit and 16-bit types, half an element for tf32 (1:2 sparsity: the indices of one chunk are
// {0,1} or {2,3}), and a pair of elements for 4-bit types (4:8 pair-wise sparsity). Every
// chunk of four units of a row keeps two units, whose 2-bit indices are bits [1:0] and [3:2]
// of a 4-bit field.
//
// Metadata layout (PTX ISA 9.7.16.6.2 figures, sparse-mma-metadata-*): with C chunks per row,
// C/4 threads of each group of four consecutive lanes supply metadata. Group g (lanes 4g..4g+3)
// serves rows g and g + 8. For 16-bit and 32-bit types the s-th supplying thread holds chunks
// 4s..4s+3, chunk 4s + j of row g in bits [4j+3:4j] and of row g + 8 in bits [16+4j+3:16+4j].
// For 8-bit and 4-bit types a supplying thread holds 8 chunks of one row: chunk 8s + j of row g
// in bits [4j+3:4j] of thread 2s and of row g + 8 in thread 2s + 1. The supplying threads are
// lane F of the group when C/4 = 1, lanes 2F and 2F+1 when C/4 = 2, and all four when C/4 = 4,
// with F the sparsity selector (operand f). tests/mma_sp.cu (fakeenv) checks these layouts on an
// RTX 5090 and in the simulator.
//
// B, C and D use the dense fragment layouts of the full shape. The products are accumulated in
// double precision and rounded once, like the generic dense model (mma_generic.cc).

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

/// Value of an element of PTX type t from its code bits (the low bits of v).
/// @param t PTX type (container already removed).
/// @param v Element bits.
/// @return Value.
double sp_elem_value(int t, uint64_t v) {
  using namespace yasai_func;
  switch (t) {
    case F16_TYPE: return mma_f16_to_f32(uint16_t(v));
    case BF16_TYPE: {
      uint32_t u = uint32_t(v & 0xFFFF) << 16;
      float f;
      memcpy(&f, &u, 4);
      return f;
    }
    case TF32_TYPE: {
      uint32_t u = uint32_t(v) & 0xFFFFE000u;  // tf32 uses the top 19 bits of the f32 pattern
      float f;
      memcpy(&f, &u, 4);
      return f;
    }
    case E4M3_TYPE: return minifloat_decode(kE4M3, unsigned(v & 0xFF));
    case E5M2_TYPE: return minifloat_decode(kE5M2, unsigned(v & 0xFF));
    case E2M1_TYPE: return minifloat_decode(kE2M1, unsigned(v & 0xF));
    case E2M3_TYPE: return minifloat_decode(kE2M3, unsigned(v & 0x3F));
    case E3M2_TYPE: return minifloat_decode(kE3M2, unsigned(v & 0x3F));
    case S8_TYPE: return double(int8_t(v & 0xFF));
    case U8_TYPE: return double(v & 0xFF);
    case S4_TYPE: return double(int(v & 7) - int(v & 8));
    case U4_TYPE: return double(v & 15);
    default:
      fprintf(stderr, "yasai-mma.sp: element type %d is not supported\n", t);
      abort();
  }
}

/// Reads the n registers of a vector operand (or a scalar register operand when n is 1).
/// @param t Thread.
/// @param op Operand.
/// @param out Values.
/// @param n Registers.
void sp_read_regs(ptx_thread_info *t, const operand_info &op, ptx_reg_t *out, unsigned n) {
  if (op.is_vector()) t->get_vector_operand_values(op, out, n);
  else out[0] = t->get_reg(op.get_symbol());
}

}  // namespace

void tensor_mma_sp_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst, int M, int N,
                        int K, unsigned tid) {
  (void)inst;
  const std::list<int> type_list = pI->get_scalar_type();
  std::vector<int> types(type_list.begin(), type_list.end());
  if (types.size() < 4) {
    fprintf(stderr, "yasai-mma.sp: mma.sp with %zu types is not supported\n", types.size());
    abort();
  }
  const int dtype = types[0], atype = types[1], btype = types[2];
  bool packed4 = false;
  for (int opt : pI->get_options())
    if (opt == TCGEN05_KIND_MXF4_OPTION || opt == TCGEN05_KIND_MXF4NVF4_OPTION) packed4 = true;
  const bool sat = pI->get_mma_saturate();
  // Container bits of an element in the A/B registers.
  auto container = [&](int t) -> unsigned {
    switch (t) {
      case F16_TYPE: case BF16_TYPE: return 16;
      case TF32_TYPE: return 32;
      case S4_TYPE: case U4_TYPE: return 4;
      case E2M1_TYPE: return packed4 ? 4 : 8;
      default: return 8;  // s8/u8, FP8, FP6 and FP4 in 8-bit containers
    }
  };
  // Code bits of an element inside its container (E2M1 in 8 bits sits in bits [5:2]).
  auto field = [&](int t, unsigned cbits, uint64_t bits) -> uint64_t {
    if (cbits == 8 && t == E2M1_TYPE) return (bits >> 2) & 0xF;
    if (cbits == 8 && (t == E2M3_TYPE || t == E3M2_TYPE)) return bits & 0x3F;
    return bits;
  };
  const unsigned abits = container(atype), bbits = container(btype);
  const unsigned pa = 32 / abits, pb = 32 / bbits;
  const int Kp = K / 2;  // packed A columns
  const unsigned na = unsigned(M * Kp) / (32 * pa), nb = unsigned(K * N) / (32 * pb);
  const bool c_f16 = (dtype == F16_TYPE);
  const unsigned nc = c_f16 ? unsigned(M * N) / 64 : unsigned(M * N) / 32;
  if (na > 8 || nb > 8 || nc > 8) {
    fprintf(stderr, "yasai-mma.sp: fragment of %u/%u/%u registers is not supported\n", na, nb, nc);
    abort();
  }
  // Units: e_per_unit elements per unit (2 for 4-bit types), u_per_e units per element (2 for tf32).
  const int e_per_unit = (abits == 4) ? 2 : 1, u_per_e = (atype == TF32_TYPE) ? 2 : 1;
  const int Ku = K * u_per_e / e_per_unit, chunks = Ku / 4, threads = chunks / 4;
  if (threads < 1 || threads > 4 || threads == 3) {
    fprintf(stderr, "yasai-mma.sp: shape m%dn%dk%d with this type is not supported\n", M, N, K);
    abort();
  }

  std::vector<double> Ap(M * Kp), A(M * K, 0.0), B(K * N), C(M * N);
  std::vector<uint32_t> meta(32);
  const operand_info &oa = pI->operand_lookup(1), &ob = pI->operand_lookup(2),
                     &oc = pI->operand_lookup(3), &oe = pI->operand_lookup(4),
                     &of = pI->operand_lookup(5);
  ptx_thread_info *t0 = core->get_thread_info()[tid];
  const unsigned sel = t0->get_operand_value(of, of, U32_TYPE, t0, 1).u32;
  for (unsigned lane = 0; lane < 32; lane++) {
    ptx_thread_info *t = core->get_thread_info()[tid + lane];
    const unsigned g = lane >> 2, q = lane & 3;
    ptx_reg_t ra[8], rb[8], rc[8];
    sp_read_regs(t, oa, ra, na);
    sp_read_regs(t, ob, rb, nb);
    sp_read_regs(t, oc, rc, nc);
    meta[lane] = t->get_operand_value(oe, oe, U32_TYPE, t, 1).u32;
    for (unsigned r = 0; r < na; r++) {
      const unsigned cw = 4 * pa, chunk = r / unsigned(M / 8), m = g + 8 * (r % unsigned(M / 8));
      for (unsigned j = 0; j < pa; j++) {
        const unsigned k = chunk * cw + q * pa + j;
        const uint64_t bits = (uint64_t(ra[r].u32) >> (j * abits)) &
                              ((abits >= 32) ? 0xFFFFFFFFull : ((1ull << abits) - 1));
        Ap[m * Kp + k] = sp_elem_value(atype, field(atype, abits, bits));
      }
    }
    for (unsigned r = 0; r < nb; r++) {
      const unsigned cw = 4 * pb;
      for (unsigned j = 0; j < pb; j++) {
        const unsigned k = r * cw + q * pb + j;
        const uint64_t bits = (uint64_t(rb[r].u32) >> (j * bbits)) &
                              ((bbits >= 32) ? 0xFFFFFFFFull : ((1ull << bbits) - 1));
        B[k * N + g] = sp_elem_value(btype, field(btype, bbits, bits));
      }
    }
    for (unsigned r = 0; r < nc; r++) {
      if (c_f16) {
        const unsigned m = g + 8 * r;
        for (unsigned j = 0; j < 2; j++)
          C[m * N + q * 2 + j] = mma_f16_to_f32(uint16_t(rc[r].u32 >> (16 * j)));
      } else {
        const unsigned m = g + 8 * (r / 2), n = q * 2 + (r % 2);
        C[m * N + n] = (dtype == S32_TYPE) ? double(rc[r].s32) : double(rc[r].f32);
      }
    }
  }
  // Decompress A with the metadata.
  const unsigned base = threads == 1 ? sel : (threads == 2 ? 2 * sel : 0);
  for (int m = 0; m < M; m++) {
    for (int c = 0; c < chunks; c++) {
      // 16/32-bit types: a thread holds 4 chunks of rows g and g + 8; 8/4-bit types: a thread
      // holds 8 chunks of row g (even thread of a pair) or g + 8 (odd).
      const bool wide = abits >= 16;
      const unsigned lane = wide ? 4 * unsigned(m % 8) + base + unsigned(c / 4)
                                 : 4 * unsigned(m % 8) + base + 2 * unsigned(c / 8) + (m >= 8 ? 1 : 0);
      const unsigned shift = wide ? 4 * unsigned(c % 4) + (m >= 8 ? 16 : 0) : 4 * unsigned(c % 8);
      const unsigned nib = (meta[lane & 31] >> shift) & 0xF, i0 = nib & 3, i1 = (nib >> 2) & 3;
      if (u_per_e == 2) {  // tf32: one element per chunk of two
        A[m * K + 2 * c + int(i0 / 2)] = Ap[m * Kp + c];
      } else if (e_per_unit == 2) {  // 4-bit pairs
        for (int h = 0; h < 2; h++) {
          A[m * K + 8 * c + 2 * int(i0) + h] = Ap[m * Kp + 4 * c + h];
          A[m * K + 8 * c + 2 * int(i1) + h] = Ap[m * Kp + 4 * c + 2 + h];
        }
      } else {
        A[m * K + 4 * c + int(i0)] = Ap[m * Kp + 2 * c];
        A[m * K + 4 * c + int(i1)] = Ap[m * Kp + 2 * c + 1];
      }
    }
  }
  const bool integer = (dtype == S32_TYPE);
  const operand_info &od = pI->operand_lookup(0);
  std::vector<ptx_reg_t> D(M * N);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      ptx_reg_t &d = D[m * N + n];
      d.u64 = 0;
      if (integer) {
        int64_t acc = int64_t(C[m * N + n]);
        for (int k = 0; k < K; k++) acc += int64_t(A[m * K + k]) * int64_t(B[k * N + n]);
        if (sat) {
          if (acc > INT32_MAX) acc = INT32_MAX;
          if (acc < INT32_MIN) acc = INT32_MIN;
        }
        d.s32 = int32_t(uint32_t(uint64_t(acc)));
        continue;
      }
      double acc = C[m * N + n];
      for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
      if (c_f16) d.u32 = yasai_func::minifloat_encode_f16(float(acc));
      else d.f32 = float(acc);
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
        out[r].u32 = (D[m * N + q * 2].u32 & 0xFFFF) | (D[m * N + q * 2 + 1].u32 << 16);
      } else {
        out[r] = D[(g + 8 * (r / 2)) * N + q * 2 + (r % 2)];
      }
    }
    if (od.is_vector()) {
      for (unsigned r = 0; r < nc; r++) t->set_reg(od.vec_symbol(r), out[r]);
    } else {
      t->set_reg(od.get_symbol(), out[0]);
    }
  }
}

}  // namespace flash_gpgpu_sim
