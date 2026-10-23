// yasai: wgmma.mma_async for every dense type combination and every N of the
// PTX ISA:
//   f32 or f16 accumulators with f16/bf16 inputs (K 16), tf32 (K 8), e4m3/e5m2
//   (K 32); s32 accumulators with s8/u8 inputs (K 32, .satfinite).
// A comes from registers (the mma.sync fragment layout of each warp's 16 rows)
// or from shared memory through a matrix descriptor; B always from shared
// memory. Transposition flags exist only for 16-bit types; the immediate scales
// negate A and B. Products are accumulated in double precision (int64 for
// integers) and rounded once to the accumulator type. The older per-type
// implementations (f16/bf16 f32, and the N = 8 forms) stay as they were;
// tensor_wgmma_impl sends every other combination here.
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
#include "../mma/tensor_mma.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

// Value of an operand element of type t from its raw bits.
double wg_value(int t, uint32_t bits) {
  switch (t) {
  case F16_TYPE:
    return mma_f16_to_f32(uint16_t(bits));
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
  case E4M3_TYPE:
    return yasai_func::minifloat_decode(yasai_func::kE4M3, bits & 0xFF);
  case E5M2_TYPE:
    return yasai_func::minifloat_decode(yasai_func::kE5M2, bits & 0xFF);
  case S8_TYPE:
    return double(int8_t(bits & 0xFF));
  case U8_TYPE:
    return double(bits & 0xFF);
  default:
    return 0;
  }
}

// Scalar type number index of the instruction (accumulator, A, B), or dflt.
int wg_type_at(const ptx_instruction *pI, unsigned index, int dflt) {
  const std::list<int> &types = pI->get_scalar_type();
  if (types.size() <= index)
    return dflt;
  auto it = types.begin();
  for (unsigned i = 0; i < index; ++i)
    ++it;
  return *it;
}

unsigned wg_bytes(int t) {
  switch (t) {
  case F16_TYPE:
  case BF16_TYPE:
    return 2;
  case TF32_TYPE:
    return 4;
  default:
    return 1;
  }
}

// f32 to f16 bits, round to nearest even.
uint16_t to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int e = int((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF)
    return uint16_t(sign | 0x7C00 | (m ? 0x200 : 0));
  if (e >= 31)
    return uint16_t(sign | 0x7C00);
  if (e <= 0) {
    if (e < -10)
      return uint16_t(sign);
    m |= 0x800000;
    const int shift = 14 - e;
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1)))
      r++;
    return uint16_t(sign | r);
  }
  uint32_t r = (uint32_t(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (r & 1)))
    r++;
  return uint16_t(sign | r);
}

} // namespace

void wgmma_generic_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst) {
  constexpr int M = 64;
  const int acc_type = wg_type_at(pI, 0, F32_TYPE);
  const int a_type = wg_type_at(pI, 1, F16_TYPE);
  const int b_type = wg_type_at(pI, 2, a_type);
  const int N = pI->get_wgmma_shape_n();
  const int K = pI->get_wgmma_shape_k();
  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &scale_d = pI->operand_lookup(3);
  const bool is_int = acc_type == S32_TYPE;
  const bool is16 = wg_bytes(a_type) == 2;
  const unsigned nops = pI->get_num_operands();
  ptx_thread_info *rep = wgmma_thread(core, inst, 0);
  const bool a_reg = src_a.is_vector();
  auto imm = [&](unsigned i, int dflt) {
    if (i >= nops)
      return dflt;
    return rep
        ->get_operand_value(pI->operand_lookup(i), pI->operand_lookup(i),
                            S32_TYPE, rep, 0)
        .s32;
  };
  int scale_a = 1, scale_b = 1, major_a = 0, major_b = 0;
  if (!is_int) {
    scale_a = imm(4, 1) == -1 ? -1 : 1;
    scale_b = imm(5, 1) == -1 ? -1 : 1;
    if (is16) {
      if (a_reg)
        major_b = imm(6, 0);
      else {
        major_a = imm(6, 0);
        major_b = imm(7, 0);
      }
    }
  }
  const uint64_t own = yasai_func::own_rank_bits();
  // B (N x K) from shared memory.
  const uint64_t b_desc =
      rep->get_operand_value(src_b, src_b, U64_TYPE, rep, 0).u64;
  const unsigned bb = wg_bytes(b_type);
  std::vector<double> B(size_t(N) * K);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const unsigned addr =
          major_b == 0 ? wgmma_gmma_k_major_smem_addr(b_desc, n, k, bb, K)
                       : wgmma_gmma_mn_major_smem_addr(b_desc, n, k, bb);
      uint32_t raw = 0;
      rep->m_shared_mem->read(addr | own, bb, &raw);
      B[size_t(n) * K + k] = scale_b * wg_value(b_type, raw);
    }
  // A (64 x K): from registers of every thread of the warp, or from shared
  // memory.
  std::vector<double> A(size_t(M) * K, 0.0);
  const unsigned ab = wg_bytes(a_type);
  const unsigned nthr = wgmma_thread_count(core, inst);
  if (a_reg) {
    for (unsigned th = 0; th < nthr; ++th) {
      ptx_thread_info *t = wgmma_thread(core, inst, th);
      const unsigned lane = wgmma_lane(t);
      const int tc = lane % 4, tr = (lane / 4) % 8, mb = lane / 32;
      ptx_reg_t regs[4];
      t->get_vector_operand_values(src_a, regs, 4);
      for (int r = 0; r < 4; ++r) {
        const int row = tr + 16 * mb + 8 * (r & 1);
        const int per = 4 / ab; // elements per register
        const int k0 =
            per == 1 ? tc + 4 * (r >> 1) : per * tc + (K / 2) * (r >> 1);
        for (int e = 0; e < per; ++e)
          A[size_t(row) * K + k0 + e] =
              scale_a * wg_value(a_type, (regs[r].u32 >> (8 * ab * e)) &
                                             ((1ull << (8 * ab)) - 1));
      }
    }
  } else {
    const uint64_t a_desc =
        rep->get_operand_value(src_a, src_a, U64_TYPE, rep, 0).u64;
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k) {
        const unsigned addr =
            major_a == 0 ? wgmma_gmma_k_major_smem_addr(a_desc, m, k, ab, K)
                         : wgmma_gmma_mn_major_smem_addr(a_desc, m, k, ab);
        uint32_t raw = 0;
        rep->m_shared_mem->read(addr | own, ab, &raw);
        A[size_t(m) * K + k] = scale_a * wg_value(a_type, raw);
      }
  }
  // Accumulate into each thread's registers.
  const bool satfinite = pI->get_wgmma_saturate();
  for (unsigned th = 0; th < nthr; ++th) {
    ptx_thread_info *t = wgmma_thread(core, inst, th);
    const unsigned lane = wgmma_lane(t);
    const bool include_c =
        (t->get_operand_value(scale_d, scale_d, PRED_TYPE, t, 0).pred & 1u) ==
        0;
    const int tc = lane % 4, tr = (lane / 4) % 8, mb = lane / 32;
    const unsigned nregs = dst.get_vect_nelem();
    for (unsigned reg = 0; reg < nregs; ++reg) {
      ptx_reg_t c = t->get_reg(dst.vec_symbol(reg));
      ptx_reg_t d;
      d.u64 = 0;
      if (acc_type == F16_TYPE) { // two packed f16 per register
        const int row = tr + 16 * mb + 8 * (reg & 1);
        uint32_t packed = 0;
        for (int h = 0; h < 2; ++h) {
          const int col = int(reg / 2) * 8 + 2 * tc + h;
          double acc =
              include_c ? mma_f16_to_f32(uint16_t(c.u32 >> (16 * h))) : 0.0;
          for (int k = 0; k < K; ++k)
            acc += A[size_t(row) * K + k] * B[size_t(col) * K + k];
          packed |= uint32_t(to_f16(float(acc))) << (16 * h);
        }
        d.u32 = packed;
      } else {
        const int row = tr + 16 * mb + 8 * ((reg >> 1) & 1);
        const int col = int(reg / 4) * 8 + 2 * tc + int(reg & 1);
        if (is_int) {
          int64_t acc = include_c ? int64_t(c.s32) : 0;
          for (int k = 0; k < K; ++k)
            acc += int64_t(A[size_t(row) * K + k]) *
                   int64_t(B[size_t(col) * K + k]);
          if (satfinite)
            acc = acc > INT32_MAX ? INT32_MAX
                                  : (acc < INT32_MIN ? INT32_MIN : acc);
          d.s32 = int32_t(acc);
        } else {
          double acc = 0;
          for (int k = 0; k < K; ++k)
            acc += A[size_t(row) * K + k] * B[size_t(col) * K + k];
          d.f32 = float(double(include_c ? c.f32 : 0.0f) + acc);
        }
      }
      t->set_reg(dst.vec_symbol(reg), d);
    }
  }
}

} // namespace flash_gpgpu_sim
