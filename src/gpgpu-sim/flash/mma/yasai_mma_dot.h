// yasai: one element of a float mma's result as a GB10 and an RTX 5090 compute it, measured on 4,096 lanes per key of mma.sync m16n8k{8,16} with f16 and bf16 operands (fakeenv
// coverage/common/ptx_key_values2.py); the PTX ISA leaves the rounding unspecified.
//   * Each nonzero product's exponent is the sum of its operands' exponents before normalization, so a product of 1.5 and 1.5 counts as exponent 0. A subnormal operand counts as its format's smallest
//     normal exponent (-14 for f16, -126 for bf16 and f32) rather than its value's. The accumulator counts with its own exponent.
//   * Every term is aligned to the largest of these exponents and cut toward zero to 25 fraction bits below it.
//   * The cut terms are added exactly, and the sum is cut toward zero to f32; with an f16 accumulator it is rounded to nearest, ties to even, to f16 (dot_f16). A zero result is +0.
//   * A NaN, or an infinity with an opposite infinity or a zero, gives the canonical NaN 0x7fffffff. A sum whose terms are all zeros gives +0.
#ifndef YASAI_MMA_DOT_H
#define YASAI_MMA_DOT_H

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace yasai_mma {

/// Fraction bits kept below the largest exponent.
constexpr int kFractionBits = 25;

/// The exponent an operand counts as.
/// @param x A nonzero finite value.
/// @param min_exp Smallest normal exponent of the operand's format.
/// @return floor(log2 |x|), at least min_exp.
inline int op_exp(double x, int min_exp) {
  const int e = std::ilogb(x);
  return e < min_exp ? min_exp : e;
}

/// The canonical NaN as a float.
inline float canonical_nan() {
  const uint32_t u = 0x7fffffffu;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

/// The result of one element: the sum over k of a(k) * b(k), plus c.
/// @param K Number of terms.
/// @param a Function from k to the operand of A, as a double (exact for every format).
/// @param b Function from k to the operand of B.
/// @param c Accumulator element.
/// @param min_exp_a Smallest normal exponent of A's format.
/// @param min_exp_b Smallest normal exponent of B's format.
/// @return The element in f32.
/// The cut sum of one element in units of 2^(emax - kFractionBits).
/// @param K Number of terms.
/// @param a Function from k to the operand of A.
/// @param b Function from k to the operand of B.
/// @param c Accumulator element.
/// @param min_exp_a Smallest normal exponent of A's format.
/// @param min_exp_b Smallest normal exponent of B's format.
/// @param min_exp_c Smallest normal exponent of the accumulator's format.
/// @param special Set when a NaN or an infinity is among the operands; then the sum is not computed.
/// @param s_special The sum in double, for special.
/// @param e0 Exponent of the sum's unit; meaningful when the result is nonzero.
/// @return The sum; zero when every term is zero or the terms cancel.
template <class FA, class FB>
inline int64_t cut_sum(int K, FA a, FB b, double c, int min_exp_a, int min_exp_b, int min_exp_c, bool *special, double *s_special, int *e0) {
  int emax = INT_MIN;
  *special = !std::isfinite(c);
  for (int k = 0; k < K; k++) {
    const double x = a(k), y = b(k);
    if (!std::isfinite(x) || !std::isfinite(y)) *special = true;
    else if (x != 0.0 && y != 0.0) {
      const int e = op_exp(x, min_exp_a) + op_exp(y, min_exp_b);
      if (e > emax) emax = e;
    }
  }
  if (*special) {
    double s = c;
    for (int k = 0; k < K; k++) s += a(k) * b(k);
    *s_special = s;
    return 0;
  }
  if (c != 0.0) {
    const int e = op_exp(c, min_exp_c);
    if (e > emax) emax = e;
  }
  if (emax == INT_MIN) return 0;
  const int scale = kFractionBits - emax;
  int64_t acc = int64_t(std::trunc(std::ldexp(c, scale)));
  for (int k = 0; k < K; k++) {
    const double t = a(k) * b(k);
    if (t != 0.0) acc += int64_t(std::trunc(std::ldexp(t, scale)));
  }
  *e0 = -scale;
  return acc;
}

/// The result of one element with an f16 accumulator: the sum over k of a(k) * b(k), plus c.
/// @param K Number of terms.
/// @param a Function from k to the operand of A, as a double.
/// @param b Function from k to the operand of B.
/// @param c Accumulator element.
/// @param min_exp_a Smallest normal exponent of A's format.
/// @param min_exp_b Smallest normal exponent of B's format.
/// @return The element's f16 bits.
template <class FA, class FB>
inline uint16_t dot_f16(int K, FA a, FB b, double c, int min_exp_a, int min_exp_b) {
  bool special = false;
  double s = 0.0;
  int e0 = 0;
  const int64_t acc = cut_sum(K, a, b, c, min_exp_a, min_exp_b, -14, &special, &s, &e0);
  if (special) return std::isnan(s) ? 0x7fffu : (s < 0.0 ? 0xfc00u : 0x7c00u);
  if (acc == 0) return 0;
  const bool neg = acc < 0;
  uint64_t mag = neg ? uint64_t(-acc) : uint64_t(acc);
  int shift = 64 - __builtin_clzll(mag) - 11;
  if (-24 - e0 > shift) shift = -24 - e0;
  if (shift > 0) {
    if (shift >= 64) return 0;
    const uint64_t low = mag & ((1ull << shift) - 1), half = 1ull << (shift - 1);
    mag >>= shift;
    e0 += shift;
    if (low > half || (low == half && (mag & 1))) mag++;
    if (mag == (1ull << 11)) {
      mag >>= 1;
      e0++;
    }
  }
  if (mag == 0) return 0;
  uint16_t bits;
  const int n = 64 - __builtin_clzll(mag);
  if (n + e0 > 16) bits = 0x7c00;
  else if (n + e0 - 1 < -14) bits = uint16_t(e0 >= -24 ? mag << (e0 + 24) : mag);
  else bits = uint16_t(((n + e0 - 1 + 15) << 10) | ((mag << (11 - n)) & 0x3ffu));
  return uint16_t(bits | (neg ? 0x8000u : 0u));
}

template <class FA, class FB>
inline float dot(int K, FA a, FB b, float c, int min_exp_a, int min_exp_b) {
  bool special = false;
  double s = 0.0;
  int e0 = 0;
  const int64_t acc = cut_sum(K, a, b, double(c), min_exp_a, min_exp_b, -126, &special, &s, &e0);
  if (special) return std::isnan(s) ? canonical_nan() : float(s);
  if (acc == 0) return 0.0f;  // terms that are all zeros or that cancel: +0 whatever their signs, on both GPUs, unlike IEEE's -0 for a sum of -0
  const bool neg = acc < 0;
  uint64_t mag = neg ? uint64_t(-acc) : uint64_t(acc);
  int shift = 64 - __builtin_clzll(mag) - 24;
  if (-149 - e0 > shift) shift = -149 - e0;
  if (shift > 0) {
    mag = shift >= 64 ? 0 : mag >> shift;
    e0 += shift;
  }
  if (mag == 0) return 0.0f;  // cut to nothing: +0 too (tf32 operands reach it)
  const double r = std::ldexp(double(mag), e0);
  const float f = r >= 0x1p128 ? float(INFINITY) : float(r);
  return neg ? -f : f;
}

}  // namespace yasai_mma

#endif  // YASAI_MMA_DOT_H
