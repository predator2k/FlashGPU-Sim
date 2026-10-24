// yasai: functional model of the sparse warpgroup MMA, wgmma.mma_async.sp (PTX ISA 9.7.17.6):
// f16/bf16 (m64nNk32), tf32 (m64nNk16), e4m3/e5m2 (m64nNk64) and s8/u8 (m64nNk64), with f32, f16
// or s32 accumulators, A from registers or shared memory and B from shared memory.
//
// Operands: d, a, b, sp-meta, sp-sel, scale-d{, imm-scale-a, imm-scale-b{, imm-trans-a,
// imm-trans-b}}. A holds the 64 x K/2 packed non-zero elements in the layout of a dense A of
// shape 64 x K/2 (registers: the dense fragment of each warp's 16 rows; shared memory: a matrix
// descriptor over K/2 columns). Each warp w supplies the metadata of rows 16w..16w+15 in the
// layout of the sparse mma.sync of the same element type (PTX ISA figures 179, 181, 183/184):
// f16/bf16 like mma.sp m16n8k32, tf32 like m16n8k16, 8-bit types like m16n8k64. The selector picks
// the supplying thread pair (16/32-bit types) or must be 0 (8-bit types). B is the dense K x N
// matrix. The products are accumulated in double precision (int64 for integers) and rounded once,
// as in wgmma_generic.cc.
#include "../../../cuda-sim/yasai_func.h"
#include "tensor_wgmma.h"

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
#include "../mma/tensor_mma.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

/// Value of an element of type t from its raw bits.
/// @param t PTX type.
/// @param bits Element bits.
/// @return Value.
double sp_wg_value(int t, uint32_t bits) {
  switch (t) {
    case F16_TYPE: return mma_f16_to_f32(uint16_t(bits));
    case BF16_TYPE: {
      uint32_t u = (bits & 0xFFFF) << 16;
      float f;
      memcpy(&f, &u, 4);
      return f;
    }
    case TF32_TYPE: {
      uint32_t u = bits & 0xFFFFE000u;
      float f;
      memcpy(&f, &u, 4);
      return f;
    }
    case E4M3_TYPE: return yasai_func::minifloat_decode(yasai_func::kE4M3, bits & 0xFF);
    case E5M2_TYPE: return yasai_func::minifloat_decode(yasai_func::kE5M2, bits & 0xFF);
    case S8_TYPE: return double(int8_t(bits & 0xFF));
    case U8_TYPE: return double(bits & 0xFF);
    default:
      fprintf(stderr, "yasai-wgmma.sp: element type %d is not supported\n", t);
      abort();
  }
}

/// Bytes of an element of type t.
/// @param t PTX type.
/// @return 2, 4 or 1.
unsigned sp_wg_bytes(int t) {
  if (t == F16_TYPE || t == BF16_TYPE) return 2;
  if (t == TF32_TYPE) return 4;
  return 1;
}

/// Scalar type at index i of the instruction (accumulator, A, B), or dflt.
/// @param pI Instruction.
/// @param i Index.
/// @param dflt Default.
/// @return Type.
int sp_wg_type_at(const ptx_instruction *pI, unsigned i, int dflt) {
  const std::list<int> &types = pI->get_scalar_type();
  if (types.size() <= i) return dflt;
  auto it = types.begin();
  for (unsigned k = 0; k < i; ++k) ++it;
  return *it;
}

/// f32 to f16 bits, round to nearest even.
/// @param f Value.
/// @return f16 bits.
uint16_t sp_to_f16(float f) { return yasai_func::minifloat_encode_f16(f); }

}  // namespace

void wgmma_sp_generic_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  constexpr int M = 64;
  const int acc_type = sp_wg_type_at(pI, 0, F32_TYPE);
  const int a_type = sp_wg_type_at(pI, 1, F16_TYPE);
  const int b_type = sp_wg_type_at(pI, 2, a_type);
  const int N = pI->get_wgmma_shape_n();
  const int K = pI->get_wgmma_shape_k();
  const int Kp = K / 2;
  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &meta_op = pI->operand_lookup(3);
  const operand_info &sel_op = pI->operand_lookup(4);
  const operand_info &scale_d = pI->operand_lookup(5);
  const bool is_int = acc_type == S32_TYPE;
  const unsigned ab = sp_wg_bytes(a_type), bb = sp_wg_bytes(b_type);
  const bool is16 = ab == 2;
  const unsigned nops = pI->get_num_operands();
  ptx_thread_info *rep = wgmma_thread(core, inst, 0);
  const bool a_reg = src_a.is_vector();
  auto imm = [&](unsigned i, int dflt) {
    if (i >= nops) return dflt;
    return rep->get_operand_value(pI->operand_lookup(i), pI->operand_lookup(i), S32_TYPE, rep, 0).s32;
  };
  int scale_a = 1, scale_b = 1, major_a = 0, major_b = 0;
  if (!is_int) {
    scale_a = imm(6, 1) == -1 ? -1 : 1;
    scale_b = imm(7, 1) == -1 ? -1 : 1;
    if (is16) {
      if (a_reg) {
        major_b = imm(8, 0);
      } else {
        major_a = imm(8, 0);
        major_b = imm(9, 0);
      }
    }
  }
  const unsigned sel = unsigned(rep->get_operand_value(sel_op, sel_op, U32_TYPE, rep, 0).u32);
  const uint64_t own = yasai_func::own_rank_bits();
  // B (N x K, dense) from shared memory.
  const uint64_t b_desc = rep->get_operand_value(src_b, src_b, U64_TYPE, rep, 0).u64;
  std::vector<double> B(size_t(N) * K);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const unsigned addr = major_b == 0 ? wgmma_gmma_k_major_smem_addr(b_desc, n, k, bb, K)
                                         : wgmma_gmma_mn_major_smem_addr(b_desc, n, k, bb);
      uint32_t raw = 0;
      rep->m_shared_mem->read(addr | own, bb, &raw);
      B[size_t(n) * K + k] = scale_b * sp_wg_value(b_type, raw);
    }
  // Packed A (64 x K/2) and the metadata of every thread.
  std::vector<double> Ap(size_t(M) * Kp, 0.0);
  std::vector<uint32_t> meta(128, 0);
  const unsigned nthr = wgmma_thread_count(core, inst);
  for (unsigned th = 0; th < nthr; ++th) {
    ptx_thread_info *t = wgmma_thread(core, inst, th);
    const unsigned lane = wgmma_lane(t);
    if (lane < 128) meta[lane] = t->get_operand_value(meta_op, meta_op, U32_TYPE, t, 0).u32;
    if (!a_reg) continue;
    const int tc = lane % 4, tr = (lane / 4) % 8, mb = lane / 32;
    ptx_reg_t regs[4];
    t->get_vector_operand_values(src_a, regs, 4);
    for (int r = 0; r < 4; ++r) {
      const int row = tr + 16 * mb + 8 * (r & 1);
      const int per = 4 / int(ab);
      const int k0 = per == 1 ? tc + 4 * (r >> 1) : per * tc + (Kp / 2) * (r >> 1);
      for (int e = 0; e < per; ++e)
        Ap[size_t(row) * Kp + k0 + e] =
            scale_a * sp_wg_value(a_type, (regs[r].u32 >> (8 * ab * e)) & ((1ull << (8 * ab)) - 1));
    }
  }
  if (!a_reg) {
    const uint64_t a_desc = rep->get_operand_value(src_a, src_a, U64_TYPE, rep, 0).u64;
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < Kp; ++k) {
        const unsigned addr = major_a == 0 ? wgmma_gmma_k_major_smem_addr(a_desc, m, k, ab, Kp)
                                           : wgmma_gmma_mn_major_smem_addr(a_desc, m, k, ab);
        uint32_t raw = 0;
        rep->m_shared_mem->read(addr | own, ab, &raw);
        Ap[size_t(m) * Kp + k] = scale_a * sp_wg_value(a_type, raw);
      }
  }
  // Decompress A with the metadata (per warp, the mma.sp layouts).
  const int u_per_e = (a_type == TF32_TYPE) ? 2 : 1;
  const int chunks = K * u_per_e / 4;  // per row
  const int nmeta = chunks / 4;        // supplying threads per group of four
  const bool wide = ab >= 2;
  const unsigned base = nmeta == 1 ? sel : (nmeta == 2 ? 2 * sel : 0);
  std::vector<double> A(size_t(M) * K, 0.0);
  for (int m = 0; m < M; ++m) {
    const int w = m / 16, rr = m % 16, g = rr % 8;
    const bool hi = rr >= 8;
    for (int c = 0; c < chunks; ++c) {
      const unsigned lane = wide ? 4 * g + base + unsigned(c / 4)
                                 : 4 * g + base + 2 * unsigned(c / 8) + (hi ? 1 : 0);
      const unsigned shift = wide ? 4 * unsigned(c % 4) + (hi ? 16 : 0) : 4 * unsigned(c % 8);
      const unsigned nib = (meta[32 * w + (lane & 31)] >> shift) & 0xF;
      const unsigned i0 = nib & 3, i1 = (nib >> 2) & 3;
      if (u_per_e == 2) {
        A[size_t(m) * K + 2 * c + int(i0 / 2)] = Ap[size_t(m) * Kp + c];
      } else {
        A[size_t(m) * K + 4 * c + int(i0)] = Ap[size_t(m) * Kp + 2 * c];
        A[size_t(m) * K + 4 * c + int(i1)] = Ap[size_t(m) * Kp + 2 * c + 1];
      }
    }
  }
  // Accumulate into each thread's registers (dense wgmma accumulator layout).
  const bool satfinite = pI->get_wgmma_saturate();
  for (unsigned th = 0; th < nthr; ++th) {
    ptx_thread_info *t = wgmma_thread(core, inst, th);
    const unsigned lane = wgmma_lane(t);
    const bool include_c = (t->get_operand_value(scale_d, scale_d, PRED_TYPE, t, 0).pred & 1u) == 0;
    const int tc = lane % 4, tr = (lane / 4) % 8, mb = lane / 32;
    const unsigned nregs = dst.get_vect_nelem();
    for (unsigned reg = 0; reg < nregs; ++reg) {
      ptx_reg_t c = t->get_reg(dst.vec_symbol(reg));
      ptx_reg_t d;
      d.u64 = 0;
      if (acc_type == F16_TYPE) {
        const int row = tr + 16 * mb + 8 * (reg & 1);
        uint32_t packed = 0;
        for (int h = 0; h < 2; ++h) {
          const int col = int(reg / 2) * 8 + 2 * tc + h;
          double acc = include_c ? mma_f16_to_f32(uint16_t(c.u32 >> (16 * h))) : 0.0;
          for (int k = 0; k < K; ++k) acc += A[size_t(row) * K + k] * B[size_t(col) * K + k];
          packed |= uint32_t(sp_to_f16(float(acc))) << (16 * h);
        }
        d.u32 = packed;
      } else {
        const int row = tr + 16 * mb + 8 * ((reg >> 1) & 1);
        const int col = int(reg / 4) * 8 + 2 * tc + int(reg & 1);
        if (is_int) {
          int64_t acc = include_c ? int64_t(c.s32) : 0;
          for (int k = 0; k < K; ++k)
            acc += int64_t(A[size_t(row) * K + k]) * int64_t(B[size_t(col) * K + k]);
          if (satfinite) acc = acc > INT32_MAX ? INT32_MAX : (acc < INT32_MIN ? INT32_MIN : acc);
          d.s32 = int32_t(acc);
        } else {
          double acc = 0;
          for (int k = 0; k < K; ++k) acc += A[size_t(row) * K + k] * B[size_t(col) * K + k];
          d.f32 = float(double(include_c ? c.f32 : 0.0f) + acc);
        }
      }
      t->set_reg(dst.vec_symbol(reg), d);
    }
  }
}

}  // namespace flash_gpgpu_sim
