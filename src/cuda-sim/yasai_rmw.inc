// yasai_rmw.inc: atomic read-modify-write on shared memory ranges, the red instruction and the
// multimem instructions (yasai fakeenv simulation backend). Included by instructions.cc after the
// atom implementation, so it uses that file's helpers (decode_space, f16/bf16 conversions).
//
// Device memory that other processes share (host_window ranges: mapped host memory, IPC and
// cuMem segments) is updated with host compare-and-swap loops, so atomics from kernels of
// different rank processes and from host threads are atomic with respect to each other. Memory of
// the simulator's own storage is updated by read, compute and write, as atom did before (the
// functional simulation thread is the only writer there).

namespace yasai_rmw {

/// Element arithmetic class of a PTX type for reductions and atomics.
enum class elem { b, u, s, f16, bf16, f32, f64 };

/// Classifies a PTX type (with the instruction's bf16x2 flag).
/// @param type PTX scalar type code (F16X2_TYPE for both packed 16-bit types).
/// @param bf16x2 Whether the packed 16-bit type is bf16x2.
/// @param bytes Receives the element size in bytes (2 for each half of a packed pair).
/// @param pairs Receives 2 for packed 16-bit pairs, 1 otherwise.
/// @return The arithmetic class.
static elem classify(unsigned type, bool bf16x2, unsigned *bytes, unsigned *pairs) {
  *pairs = 1;
  switch (type) {
    case U32_TYPE: *bytes = 4; return elem::u;
    case S32_TYPE: *bytes = 4; return elem::s;
    case B32_TYPE: *bytes = 4; return elem::b;
    case U64_TYPE: *bytes = 8; return elem::u;
    case S64_TYPE: *bytes = 8; return elem::s;
    case B64_TYPE: *bytes = 8; return elem::b;
    case F32_TYPE: *bytes = 4; return elem::f32;
    case F64_TYPE: *bytes = 8; return elem::f64;
    case F16_TYPE: *bytes = 2; return elem::f16;
    case BF16_TYPE: *bytes = 2; return elem::bf16;
    case F16X2_TYPE: *bytes = 2; *pairs = 2; return bf16x2 ? elem::bf16 : elem::f16;
    case U16_TYPE: case B16_TYPE: *bytes = 2; return elem::u;
    case S16_TYPE: *bytes = 2; return elem::s;
    default: *bytes = 4; return elem::b;
  }
}

/// Float value of an element's bits.
/// @param k Arithmetic class (f16, bf16, f32).
/// @param bits Element bits.
/// @return Value.
static double to_f(elem k, uint64_t bits) {
  switch (k) {
    case elem::f16: return f32_from_f16_bits(uint16_t(bits));
    case elem::bf16: return f32_from_bf16_bits(uint16_t(bits));
    case elem::f32: { uint32_t u = uint32_t(bits); float f; memcpy(&f, &u, 4); return f; }
    default: { double d; memcpy(&d, &bits, 8); return d; }
  }
}

/// Element bits of a float value (round to nearest even).
/// @param k Arithmetic class (f16, bf16, f32, f64).
/// @param v Value.
/// @return Element bits.
static uint64_t from_f(elem k, double v) {
  switch (k) {
    case elem::f16: return f16_bits_from_f32(float(v), RN_OPTION);
    case elem::bf16: return bf16_bits_from_f32(float(v), RN_OPTION);
    case elem::f32: { float f = float(v); uint32_t u; memcpy(&u, &f, 4); return u; }
    default: { uint64_t u; memcpy(&u, &v, 8); return u; }
  }
}

/// Sign-extends an element of `bytes` bytes.
/// @param v Element bits.
/// @param bytes Element size.
/// @return Signed value.
static int64_t sext(uint64_t v, unsigned bytes) {
  const unsigned sh = 64 - 8 * bytes;
  return int64_t(v << sh) >> sh;
}

/// Mask of an element of `bytes` bytes.
/// @param bytes Element size.
/// @return Mask.
static uint64_t mask_of(unsigned bytes) { return bytes >= 8 ? ~0ull : ((1ull << (8 * bytes)) - 1); }

/// New value of one element for atomic or reduction operation `op`.
/// @param op ATOMIC_* code.
/// @param k Arithmetic class.
/// @param bytes Element size.
/// @param a Current element bits.
/// @param b Operand b bits.
/// @param c Operand c bits (CAS new value).
/// @return New element bits.
static uint64_t apply1(unsigned op, elem k, unsigned bytes, uint64_t a, uint64_t b, uint64_t c) {
  const uint64_t m = mask_of(bytes);
  a &= m; b &= m; c &= m;
  const bool fp = k == elem::f16 || k == elem::bf16 || k == elem::f32 || k == elem::f64;
  switch (op) {
    case ATOMIC_ADD:
      if (fp) return from_f(k, to_f(k, a) + to_f(k, b)) & m;
      return (a + b) & m;
    case ATOMIC_MIN:
      if (fp) return from_f(k, std::fmin(to_f(k, a), to_f(k, b))) & m;
      if (k == elem::s) return sext(a, bytes) < sext(b, bytes) ? a : b;
      return a < b ? a : b;
    case ATOMIC_MAX:
      if (fp) return from_f(k, std::fmax(to_f(k, a), to_f(k, b))) & m;
      if (k == elem::s) return sext(a, bytes) > sext(b, bytes) ? a : b;
      return a > b ? a : b;
    case ATOMIC_AND: return a & b;
    case ATOMIC_OR: return a | b;
    case ATOMIC_XOR: return a ^ b;
    case ATOMIC_INC: return (a >= b) ? 0 : (a + 1) & m;
    case ATOMIC_DEC: return (a == 0 || a > b) ? b : (a - 1) & m;
    case ATOMIC_EXCH: return b;
    case ATOMIC_CAS: return a == b ? c : a;
    default:
      printf("yasai_rmw: unsupported atomic operation %u\n", op);
      assert(0);
      return a;
  }
}

/// New value of a whole word (one element or a packed pair).
/// @param op ATOMIC_* code.
/// @param type PTX type.
/// @param bf16x2 Packed bf16 flag.
/// @param a Current word.
/// @param b Operand b.
/// @param c Operand c.
/// @param word_bytes Receives the word size.
/// @return New word.
static uint64_t apply(unsigned op, unsigned type, bool bf16x2, uint64_t a, uint64_t b, uint64_t c,
                      unsigned *word_bytes) {
  unsigned eb, pairs;
  elem k = classify(type, bf16x2, &eb, &pairs);
  *word_bytes = eb * pairs;
  if (pairs == 1) return apply1(op, k, eb, a, b, c);
  uint64_t lo = apply1(op, k, eb, a & 0xffff, b & 0xffff, c & 0xffff);
  uint64_t hi = apply1(op, k, eb, (a >> 16) & 0xffff, (b >> 16) & 0xffff, (c >> 16) & 0xffff);
  return lo | (hi << 16);
}

/// Whether an access goes to a shared (identity mapped) host_window range.
/// @param space Memory space of the access.
/// @param addr Address.
/// @param bytes Size.
/// @return True for global accesses inside a host or device range.
static bool shared_word(memory_space_t space, addr_t addr, unsigned bytes) {
  if (space != global_space) return false;
  host_window::range_info r;
  if (!host_window::lookup(addr, bytes, &r)) return false;
  return r.kind != host_window::range_kind::multicast && (addr % bytes) == 0 &&
         (bytes == 2 || bytes == 4 || bytes == 8);
}

/// Atomic read-modify-write of the word at host address `addr` with a compare-and-swap loop.
/// @param addr Host address (naturally aligned).
/// @param bytes Word size (2, 4 or 8).
/// @param op ATOMIC_* code.
/// @param type PTX type.
/// @param bf16x2 Packed bf16 flag.
/// @param b Operand b.
/// @param c Operand c.
/// @return The word's previous value.
static uint64_t host_rmw(uint64_t addr, unsigned bytes, unsigned op, unsigned type, bool bf16x2,
                         uint64_t b, uint64_t c) {
  uint64_t old = 0;
  host_window::read(addr, bytes, &old);
  for (;;) {
    unsigned wb;
    uint64_t nv = apply(op, type, bf16x2, old, b, c, &wb);
    uint64_t expect = old;
    if (host_window::compare_exchange(addr, bytes, &expect, nv)) return old;
    old = expect;
  }
}

/// One atomic read-modify-write in any memory space.
/// @param thread Executing thread (for watchpoints).
/// @param pI Instruction.
/// @param mem Memory space object of the access.
/// @param space Memory space.
/// @param addr Address in that space.
/// @param op ATOMIC_* code.
/// @param type PTX type.
/// @param b Operand b.
/// @param c Operand c.
/// @return The previous value.
static uint64_t rmw(ptx_thread_info *thread, const ptx_instruction *pI, memory_space *mem,
                    memory_space_t space, addr_t addr, unsigned op, unsigned type, uint64_t b,
                    uint64_t c) {
  unsigned eb, pairs;
  classify(type, pI->is_bf16x2(), &eb, &pairs);
  const unsigned bytes = eb * pairs;
  if (shared_word(space, addr, bytes))
    return host_rmw(addr, bytes, op, type, pI->is_bf16x2(), b, c);
  uint64_t old = 0;
  mem->read(addr, bytes, &old);
  unsigned wb;
  uint64_t nv = apply(op, type, pI->is_bf16x2(), old, b, c, &wb);
  mem->write(addr, bytes, &nv, thread, pI);
  return old;
}

/// Resolves the address operand of red/multimem to a memory space and address.
/// @param pI Instruction.
/// @param thread Executing thread.
/// @param op Address operand.
/// @param space In: the instruction's space; out: the resolved space.
/// @param mem Receives the memory space object.
/// @return Address in that space.
static addr_t resolve(const ptx_instruction *pI, ptx_thread_info *thread, const operand_info &op,
                      memory_space_t &space, memory_space *&mem) {
  addr_t addr = thread->get_operand_value(op, op, U64_TYPE, thread, 1).u64;
  if (space == undefined_space) {
    if (whichspace(addr) == shared_space) {
      addr = generic_to_shared(thread->get_hw_sid(), addr);
      space = shared_space;
    } else {
      space = global_space;
    }
  }
  if (space == shared_space) {
    mem = thread->m_shared_mem;
    addr &= 0x00000000FFFFFFFFULL;
  } else {
    mem = thread->get_global_memory();
  }
  (void)pI;
  return addr;
}

/// Number of vector elements of an instruction (1 without a vector type).
/// @param pI Instruction.
/// @return 1, 2, 3, 4 or 8.
static unsigned vec_count(const ptx_instruction *pI) {
  switch (pI->get_vector()) {
    case V2_TYPE: return 2;
    case V3_TYPE: return 3;
    case V4_TYPE: return 4;
    case V8_TYPE: return 8;
    default: return 1;
  }
}

/// Reads a source operand as `n` 64-bit words (scalar or vector of registers).
/// @param thread Executing thread.
/// @param src Source operand.
/// @param ref Operand that sets the register type.
/// @param type PTX type.
/// @param n Elements.
/// @param out Receives the values.
static void get_src(ptx_thread_info *thread, const operand_info &src, const operand_info &ref,
                    unsigned type, unsigned n, ptx_reg_t *out) {
  if (src.is_vector()) {
    thread->get_vector_operand_values(src, out, n);
  } else {
    out[0] = thread->get_operand_value(src, ref, type, thread, 1);
  }
}

}  // namespace yasai_rmw

/// red{.sem}{.scope}{.space}.op{.vec}.type [a], b: atomic reduction without a result.
void red_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  memory_space_t space = pI->get_space();
  memory_space *mem = NULL;
  addr_t addr = yasai_rmw::resolve(pI, thread, dst, space, mem);
  const unsigned type = pI->get_type();
  const unsigned n = yasai_rmw::vec_count(pI);
  ptx_reg_t v[8];
  yasai_rmw::get_src(thread, src1, dst, type, n, v);
  unsigned eb, pairs;
  yasai_rmw::classify(type, pI->is_bf16x2(), &eb, &pairs);
  for (unsigned e = 0; e < n; e++)
    yasai_rmw::rmw(thread, pI, mem, space, addr + e * eb * pairs, pI->get_atomic(), type,
                   v[e].u64, 0);
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
}

/// Atom on a shared host_window word: performed with a host compare-and-swap loop so that it is
/// atomic with respect to other processes. Called from atom_callback.
/// @param pI Instruction.
/// @param thread Executing thread.
/// @param addr Global address.
/// @param b Operand b value.
/// @param c Operand c value (CAS).
/// @return The previous value.
uint64_t yasai_atom_shared(const ptx_instruction *pI, ptx_thread_info *thread, addr_t addr,
                           uint64_t b, uint64_t c) {
  unsigned eb, pairs;
  yasai_rmw::classify(pI->get_type(), pI->is_bf16x2(), &eb, &pairs);
  (void)thread;
  return yasai_rmw::host_rmw(addr, eb * pairs, pI->get_atomic(), pI->get_type(), pI->is_bf16x2(),
                             b, c);
}

/// Whether an atom in `space` at `addr` must use yasai_atom_shared.
/// @param pI Instruction.
/// @param space Resolved memory space.
/// @param addr Address.
/// @return True for shared host_window words.
bool yasai_atom_is_shared(const ptx_instruction *pI, memory_space_t space, addr_t addr) {
  unsigned eb, pairs;
  yasai_rmw::classify(pI->get_type(), pI->is_bf16x2(), &eb, &pairs);
  return yasai_rmw::shared_word(space, addr, eb * pairs);
}

/// Multicast object and offset of a multicast address.
/// @param addr Address.
/// @param bytes Access size.
/// @param mc Receives the multicast object.
/// @param off Receives the offset.
/// @return True when addr lies in a multicast range.
static bool yasai_mc_of(addr_t addr, size_t bytes, void **mc, uint64_t *off) {
  host_window::range_info r;
  if (!host_window::lookup(addr, bytes, &r) || r.kind != host_window::range_kind::multicast)
    return false;
  *mc = r.mc;
  *off = r.mc_offset + (addr - r.base);
  return true;
}

/// multimem.ld_reduce{.ldsem}{.scope}{.ss}.op{.acc}{.vec}.type d, [a]: loads the element from
/// every device bound to the multicast address and reduces (16-bit floats accumulate in f32).
void multimem_ld_reduce_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  addr_t addr = thread->get_operand_value(src1, src1, U64_TYPE, thread, 1).u64;
  const unsigned type = pI->get_type();
  const unsigned n = yasai_rmw::vec_count(pI);
  unsigned eb, pairs;
  yasai_rmw::elem k = yasai_rmw::classify(type, pI->is_bf16x2(), &eb, &pairs);
  const unsigned wb = eb * pairs;
  void *mc;
  uint64_t off;
  if (!yasai_mc_of(addr, size_t(wb) * n, &mc, &off)) {
    printf("multimem.ld_reduce: address 0x%llx is not a multicast address\n",
           (unsigned long long)addr);
    abort();
  }
  uint64_t tg[shared_devmem::kMaxMcDevices];
  const int nt = shared_devmem::mc_targets(mc, off, size_t(wb) * n, tg, shared_devmem::kMaxMcDevices);
  const bool fp = k == yasai_rmw::elem::f16 || k == yasai_rmw::elem::bf16 ||
                  k == yasai_rmw::elem::f32 || k == yasai_rmw::elem::f64;
  const unsigned op = pI->get_atomic();
  // YASAI_MC_DEBUG=1: prints the targets and the values of 8-byte integer reductions (the
  // step counters NCCL's NVLS kernels synchronize on).
  static const bool mc_debug = getenv("YASAI_MC_DEBUG") != NULL;
  static int mc_debug_data = 0;
  if (mc_debug && fp && thread->get_laneid() == 0 && mc_debug_data < 64) {
    mc_debug_data++;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    printf("yasai-mc: t %lld.%06ld data ld_reduce off 0x%llx targets %d:", (long long)ts.tv_sec,
           ts.tv_nsec / 1000, (unsigned long long)off, nt);
    for (int t = 0; t < nt; t++) {
      uint32_t v = 0;
      host_window::read(tg[t], 4, &v);
      float f;
      memcpy(&f, &v, 4);
      printf(" %g", f);
    }
    printf("\n");
  }
  if (mc_debug && eb == 8 && !fp && thread->get_laneid() == 0) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    printf("yasai-mc: t %lld.%06ld ld_reduce op %u off 0x%llx targets %d:", (long long)ts.tv_sec,
           ts.tv_nsec / 1000, op, (unsigned long long)off, nt);
    for (int t = 0; t < nt; t++) {
      uint64_t v = 0;
      host_window::read(tg[t], 8, &v);
      printf(" %llu", (unsigned long long)v);
    }
    printf("\n");
  }
  ptx_reg_t out[8];
  for (unsigned e = 0; e < n; e++) {
    out[e].u64 = 0;
    for (unsigned h = 0; h < pairs; h++) {
      const uint64_t eoff = uint64_t(e) * wb + uint64_t(h) * eb;
      uint64_t r = 0;
      double acc = 0;
      for (int t = 0; t < nt; t++) {
        uint64_t v = 0;
        host_window::read(tg[t] + eoff, eb, &v);
        if (fp) {
          double x = yasai_rmw::to_f(k, v);
          if (t == 0) acc = x;
          else if (op == ATOMIC_ADD) acc += x;
          else if (op == ATOMIC_MIN) acc = std::fmin(acc, x);
          else if (op == ATOMIC_MAX) acc = std::fmax(acc, x);
        } else {
          r = (t == 0) ? v : yasai_rmw::apply1(op, k, eb, r, v, 0);
        }
      }
      if (fp) r = yasai_rmw::from_f(k, acc);
      out[e].u64 |= (r & yasai_rmw::mask_of(eb)) << (16 * h);
    }
  }
  if (n == 1) {
    if (dst.is_vector()) thread->set_vector_operand_values(dst, out[0], out[0], out[0], out[0]);
    else thread->set_operand_value(dst, out[0], type, thread, pI);
  } else if (n == 8) {
    for (unsigned e = 0; e < 8 && e < dst.get_vect_nelem(); e++)
      thread->set_reg(dst.vec_symbol(e), out[e]);
  } else {
    thread->set_vector_operand_values(dst, out[0], out[n > 1 ? 1 : 0], out[n > 2 ? 2 : 0],
                                      out[n > 3 ? 3 : 0]);
  }
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = global_space;
}

/// multimem.st{.stsem}{.scope}{.ss}{.vec}.type [a], b: stores to every device bound to the
/// multicast address.
void multimem_st_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  addr_t addr = thread->get_operand_value(dst, dst, U64_TYPE, thread, 1).u64;
  const unsigned type = pI->get_type();
  const unsigned n = yasai_rmw::vec_count(pI);
  unsigned eb, pairs;
  yasai_rmw::classify(type, pI->is_bf16x2(), &eb, &pairs);
  const unsigned wb = eb * pairs;
  ptx_reg_t v[8];
  yasai_rmw::get_src(thread, src1, dst, type, n, v);
  void *mc;
  uint64_t off;
  if (!yasai_mc_of(addr, size_t(wb) * n, &mc, &off)) {
    printf("multimem.st: address 0x%llx is not a multicast address\n", (unsigned long long)addr);
    abort();
  }
  uint64_t tg[shared_devmem::kMaxMcDevices];
  const int nt = shared_devmem::mc_targets(mc, off, size_t(wb) * n, tg, shared_devmem::kMaxMcDevices);
  for (int t = 0; t < nt; t++)
    for (unsigned e = 0; e < n; e++) host_window::write(tg[t] + uint64_t(e) * wb, wb, &v[e].u64);
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = global_space;
}

/// multimem.red{.redsem}{.scope}{.ss}.op{.vec}.type [a], b: atomic reduction on every device bound
/// to the multicast address.
void multimem_red_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  addr_t addr = thread->get_operand_value(dst, dst, U64_TYPE, thread, 1).u64;
  const unsigned type = pI->get_type();
  const unsigned n = yasai_rmw::vec_count(pI);
  unsigned eb, pairs;
  yasai_rmw::classify(type, pI->is_bf16x2(), &eb, &pairs);
  const unsigned wb = eb * pairs;
  ptx_reg_t v[8];
  yasai_rmw::get_src(thread, src1, dst, type, n, v);
  void *mc;
  uint64_t off;
  if (!yasai_mc_of(addr, size_t(wb) * n, &mc, &off)) {
    printf("multimem.red: address 0x%llx is not a multicast address\n", (unsigned long long)addr);
    abort();
  }
  uint64_t tg[shared_devmem::kMaxMcDevices];
  const int nt = shared_devmem::mc_targets(mc, off, size_t(wb) * n, tg, shared_devmem::kMaxMcDevices);
  for (int t = 0; t < nt; t++)
    for (unsigned e = 0; e < n; e++)
      yasai_rmw::host_rmw(tg[t] + uint64_t(e) * wb, wb, pI->get_atomic(), type, pI->is_bf16x2(),
                          v[e].u64, 0);
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = global_space;
}
