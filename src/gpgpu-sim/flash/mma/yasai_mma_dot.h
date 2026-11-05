// yasai: one element of a float mma's result as a GB10 and an RTX 5090 compute it, measured on 4,096 lanes per key of mma.sync m16n8k{8,16} with f16 and bf16 operands (fakeenv
// coverage/common/ptx_key_values2.py); the PTX ISA leaves the rounding unspecified.
//   * Each nonzero product's exponent is the sum of its operands' exponents before normalization, so a product of 1.5 and 1.5 counts as exponent 0. A subnormal operand counts as its format's smallest
//     normal exponent (-14 for f16, -126 for bf16 and f32) rather than its value's. The accumulator counts with its own exponent.
//   * Every term is aligned to the largest of these exponents and cut toward zero to 25 fraction bits below it.
//   * The cut terms are added exactly, and the sum is cut toward zero to f32.
//   * A NaN, or an infinity with an opposite infinity or a zero, gives the canonical NaN 0x7fffffff.
// On the keys above this model gives every non-NaN, finite result bit for bit (0 of 13,175 words differ).
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
template <class FA, class FB>
inline float dot(int K, FA a, FB b, float c, int min_exp_a, int min_exp_b) {
  int emax = INT_MIN;
  bool special = !std::isfinite(c);
  for (int k = 0; k < K; k++) {
    const double x = a(k), y = b(k);
    if (!std::isfinite(x) || !std::isfinite(y)) special = true;
    else if (x != 0.0 && y != 0.0) {
      const int e = op_exp(x, min_exp_a) + op_exp(y, min_exp_b);
      if (e > emax) emax = e;
    }
  }
  if (special) {
    double s = double(c);
    for (int k = 0; k < K; k++) s += a(k) * b(k);
    return std::isnan(s) ? canonical_nan() : float(s);
  }
  if (c != 0.0f) {
    const int e = op_exp(double(c), -126);
    if (e > emax) emax = e;
  }
  if (emax == INT_MIN) {  // every term is a zero: their IEEE sum
    double s = double(c);
    for (int k = 0; k < K; k++) s += a(k) * b(k);
    return float(s);
  }
  const int scale = kFractionBits - emax;
  int64_t acc = int64_t(std::trunc(std::ldexp(double(c), scale)));
  for (int k = 0; k < K; k++) {
    const double t = a(k) * b(k);
    if (t != 0.0) acc += int64_t(std::trunc(std::ldexp(t, scale)));
  }
  if (acc == 0) return 0.0f;
  const bool neg = acc < 0;
  uint64_t mag = neg ? uint64_t(-acc) : uint64_t(acc);
  int e0 = -scale;
  int shift = 64 - __builtin_clzll(mag) - 24;
  if (-149 - e0 > shift) shift = -149 - e0;
  if (shift > 0) {
    mag = shift >= 64 ? 0 : mag >> shift;
    e0 += shift;
  }
  const double r = std::ldexp(double(mag), e0);
  const float f = r >= 0x1p128 ? float(INFINITY) : float(r);
  return neg ? -f : f;
}

}  // namespace yasai_mma

#endif  // YASAI_MMA_DOT_H
