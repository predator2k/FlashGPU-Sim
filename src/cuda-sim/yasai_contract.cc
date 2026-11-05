// yasai_contract.cc: see yasai_contract.h.
#include "yasai_contract.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

#include "opcodes.h"
#include "ptx_ir.h"
#include "ptx_parser.h"
typedef void *yyscan_t;  // ptx.tab.h needs it (as ptx_ir.cc declares it)
#include "ptx.tab.h"

namespace yasai_contract {

bool enabled() {
  static const bool on = [] {
    const char *v = getenv("YASAI_FMA_CONTRACT");
    if (v != nullptr && strcmp(v, "ptxas") == 0) return true;
    if (v != nullptr && strcmp(v, "off") == 0) return false;
    fprintf(stderr,
            "yasai-contract: YASAI_FMA_CONTRACT must be ptxas (contract a mul and an add without "
            "rounding modifiers into one fused multiply-add, as ptxas does) or off; it is %s%s%s\n",
            v ? "\"" : "unset", v ? v : "", v ? "\"" : "");
    exit(2);
  }();
  return on;
}

namespace {

/// Whether t is a type that ptxas contracts (yasai_contract.h); bf16x2 is F16X2_TYPE with the
/// instruction's bf16x2 flag.
bool float_type(unsigned t) {
  return t == F32_TYPE || t == F64_TYPE || t == FF64_TYPE || t == F16_TYPE || t == BF16_TYPE ||
         t == F16X2_TYPE || t == F32X2_TYPE;
}

/// Whether the instruction ends a basic block (or keeps a product from crossing it).
bool ends_block(const ptx_instruction *p) {
  switch (p->get_opcode()) {
    case BRA_OP: case BRX_OP: case CALL_OP: case CALLP_OP: case RET_OP: case RETP_OP:
    case EXIT_OP: case BAR_OP: case BREAK_OP: case BREAKADDR_OP:
      return true;
    default:
      return false;
  }
}

/// The symbols that operand op names (as names() matches them), appended to out.
void named(const operand_info &op, std::vector<const symbol *> *out) {
  if (!op.is_valid()) return;
  if (op.is_vector()) {
    for (unsigned i = 0; i < op.get_vect_nelem(); i++)
      if (op.vec_symbol_or_null(i) != nullptr) out->push_back(op.vec_symbol_or_null(i));
    return;
  }
  if (op.is_literal()) return;
  switch (op.get_type()) {
    case symbolic_t: case reg_t: case memory_t: case address_t:
      out->push_back(op.get_symbol());
      return;
    default:
      return;
  }
}

/// Whether operand op names register r (as a register, a vector element or a memory base).
bool names(const operand_info &op, const symbol *r) {
  std::vector<const symbol *> v;
  named(op, &v);
  return std::find(v.begin(), v.end(), r) != v.end();
}

/// Whether the instruction's first operand is a destination: its registers are written, not read.
bool first_is_destination(const ptx_instruction *p) {
  if (p->get_num_operands() == 0) return false;
  switch (p->get_opcode()) {
    case ST_OP: case RED_OP: case BRA_OP: case BRX_OP: case CALL_OP: case CALLP_OP:
      return false;
    default:
      break;
  }
  return !p->operand_lookup(0).is_memory_operand();
}

/// Sorts v and removes repeated entries.
void unique_sorted(std::vector<const symbol *> *v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

/// Per function: the instructions that read and write each register, and the next block boundary
/// after each position, so the analysis is linear in the function's length.
struct function_index {
  /// Register -> the indices (ascending) of the instructions that read it (any operand other than
  /// a destination, and the guard predicate); labels never.
  std::unordered_map<const symbol *, std::vector<size_t>> readers;
  /// Register -> the indices (ascending) of the instructions that write it through their first
  /// operand.
  std::unordered_map<const symbol *, std::vector<size_t>> writers;
  /// boundary[i]: the first index after i that is a label or ends a block, or the length.
  std::vector<size_t> boundary;

  explicit function_index(const std::vector<ptx_instruction *> &insts) {
    const size_t n = insts.size();
    std::vector<const symbol *> rd, wr;
    for (size_t j = 0; j < n; j++) {
      const ptx_instruction *p = insts[j];
      if (p->is_label()) continue;
      rd.clear();
      wr.clear();
      const unsigned nop = p->get_num_operands();
      const bool dst = first_is_destination(p);
      for (unsigned i = 0; i < nop; i++) named(p->operand_lookup(i), (i == 0 && dst) ? &wr : &rd);
      if (p->has_pred()) named(p->get_pred(), &rd);
      unique_sorted(&rd);
      unique_sorted(&wr);
      // A register written by the first operand and named by another operand is read as well.
      for (const symbol *r : rd) readers[r].push_back(j);
      for (const symbol *r : wr) writers[r].push_back(j);
    }
    boundary.assign(n, n);
    for (size_t i = n; i-- > 1;) {
      const ptx_instruction *q = insts[i];
      boundary[i - 1] = (q->is_label() || ends_block(q)) ? i : boundary[i];
    }
  }

  /// The end (exclusive) of the window of a definition of r at position i: up to the next label
  /// or block end, or through the next definition of r (it may read r as well; later instructions
  /// see the new value).
  size_t window_end(size_t i, const symbol *r) const {
    const size_t b = boundary[i];
    const auto w = writers.find(r);
    if (w != writers.end()) {
      const auto d = std::upper_bound(w->second.begin(), w->second.end(), i);
      if (d != w->second.end() && *d < b) return *d + 1;
    }
    return b;
  }

  /// The instructions of (def, to) that read r, or false when r is read outside that window or by
  /// an instruction that ok does not accept.
  template <class Ok>
  bool uses_in_window(const std::vector<ptx_instruction *> &insts, size_t def, size_t to,
                      const symbol *r, Ok ok, std::vector<size_t> *uses) const {
    const auto it = readers.find(r);
    if (it == readers.end()) return false;
    for (size_t j : it->second) {
      if (j == def) continue;
      if (j <= def || j >= to) return false;  // a use outside the block window
      if (!ok(j, insts[j])) return false;
      uses->push_back(j);
    }
    return !uses->empty();
  }
};

/// Whether two instructions agree in type, bf16x2 flag and .ftz, as a contracted pair must.
bool same_kind(const ptx_instruction *m, const ptx_instruction *a) {
  return m->get_type() == a->get_type() && m->is_bf16x2() == a->is_bf16x2() &&
         m->is_ftz() == a->is_ftz();
}

/// Whether the rounding modifiers allow contraction (none, or .rn on f32x2).
bool rounding_ok(const ptx_instruction *p) {
  return !p->has_rounding_modifier() ||
         (p->get_type() == F32X2_TYPE && p->rounding_mode() == RN_OPTION);
}

/// Whether p is an add or sub that may take a contracted product of mul m.
bool fusable_add(const ptx_instruction *p, const ptx_instruction *m) {
  return (p->get_opcode() == ADD_OP || p->get_opcode() == SUB_OP) && !p->has_pred() &&
         same_kind(m, p) && rounding_ok(p) && p->get_num_operands() == 3;
}

}  // namespace

unsigned analyze(const std::list<ptx_instruction *> &body, symbol_table *symtab,
                 const char *filename) {
  (void)symtab;
  (void)filename;
  if (!enabled()) return 0;
  const std::vector<ptx_instruction *> insts(body.begin(), body.end());
  const function_index index(insts);
  unsigned marked = 0;
  // Candidate products per add: source position (1 or 2) -> (mul index, negated).
  struct prod {
    size_t mul;
    bool neg;
  };
  std::map<size_t, std::map<int, prod>> cand;
  for (size_t i = 0; i < insts.size(); i++) {
    ptx_instruction *m = insts[i];
    if (m->is_label() || m->get_opcode() != MUL_OP || m->has_pred() || m->saturation_mode())
      continue;
    if (!float_type(m->get_type()) || !rounding_ok(m) || m->get_num_operands() != 3) continue;
    const operand_info &d = m->dst();
    if (!d.is_reg() || d.is_vector()) continue;
    const symbol *r = d.get_symbol();
    // The block window: up to the next label or block end, or the next definition of r.
    const size_t to = index.window_end(i, r);
    std::vector<size_t> uses;
    bool ok = index.uses_in_window(insts, i, to, r, [&](size_t, const ptx_instruction *p) {
      if (fusable_add(p, m)) return true;
      return p->get_opcode() == NEG_OP && same_kind(m, p) && !p->has_pred() && !p->is_bf16x2();
    }, &uses);
    if (!ok) continue;
    // A neg's result must itself be used only by such adds in the window.
    struct use {
      size_t add;         // the add or sub
      const symbol *reg;  // the register it reads: r, or the neg's result
      bool neg;           // the product is negated
    };
    std::vector<use> adds;
    for (size_t u : uses) {
      const ptx_instruction *p = insts[u];
      if (p->get_opcode() != NEG_OP) {
        adds.push_back(use{u, r, false});
        continue;
      }
      const operand_info &nd = p->dst();
      if (!nd.is_reg() || nd.is_vector()) {
        ok = false;
        break;
      }
      std::vector<size_t> nuses;
      if (!index.uses_in_window(insts, u, to, nd.get_symbol(),
                                [&](size_t, const ptx_instruction *q) { return fusable_add(q, m); },
                                &nuses)) {
        ok = false;
        break;
      }
      for (size_t v : nuses) adds.push_back(use{v, nd.get_symbol(), true});
    }
    if (!ok) continue;
    for (const use &a : adds) {
      const ptx_instruction *p = insts[a.add];
      for (int pos = 1; pos <= 2; pos++) {
        const operand_info &op = p->operand_lookup(pos);
        if (!op.is_vector() && names(op, a.reg)) cand[a.add][pos] = prod{i, a.neg};
      }
    }
  }
  // Per add: the product at the first source position, else at the second.
  std::map<size_t, int> ids;  // mul -> pair id
  for (const auto &c : cand) {
    ptx_instruction *a = insts[c.first];
    const auto it = c.second.count(1) ? c.second.find(1) : c.second.find(2);
    const prod &pr = it->second;
    auto s = ids.find(pr.mul);
    if (s == ids.end()) {
      s = ids.emplace(pr.mul, int(ids.size())).first;
      insts[pr.mul]->yasai_set_save_id(s->second);
    }
    a->yasai_set_fuse(it->first, pr.neg, s->second);
    marked++;
  }
  return marked;
}

unsigned assign_slots(const std::list<ptx_instruction *> &insts, const char *func) {
  // The adds that still fuse each pair (a rewrite that replaces an add drops its marks).
  std::unordered_map<int, unsigned> pending;
  for (const ptx_instruction *p : insts)
    if (p != nullptr && p->yasai_fuse_src() != 0) pending[p->yasai_fuse_id()]++;
  if (pending.empty()) return 0;  // no pair (or YASAI_FMA_CONTRACT=off): no mul saves
  std::unordered_map<int, int> slot_of;  // pair id -> slot, while busy
  std::vector<int> free_slots;           // free slots, the lowest last
  unsigned slots = 0;
  const auto stop = [&](const ptx_instruction *p, const char *what) {
    fprintf(stderr, "yasai-contract: function %s: %s at %s:%u (%s)\n", func, what,
            p->source_file(), p->source_line(), p->get_source());
    exit(2);
  };
  for (ptx_instruction *p : insts) {
    if (p == nullptr) continue;
    if (p->is_label() || ends_block(p)) {
      if (!slot_of.empty()) stop(p, "a contracted mul and its fused add lie on two sides of this");
      continue;
    }
    if (p->yasai_fuse_src() != 0) {
      const auto s = slot_of.find(p->yasai_fuse_id());
      if (s == slot_of.end()) stop(p, "this fused add comes before its mul");
      p->yasai_set_fuse_slot(s->second);
      if (--pending[p->yasai_fuse_id()] == 0) {
        free_slots.push_back(s->second);
        std::sort(free_slots.begin(), free_slots.end(), std::greater<int>());
        slot_of.erase(s);
      }
    }
    if (p->yasai_save_id() >= 0) {
      const auto n = pending.find(p->yasai_save_id());
      if (n == pending.end() || n->second == 0) {
        p->yasai_set_save_slot(-1);
        continue;
      }
      int slot;
      if (free_slots.empty()) {
        slot = int(slots++);
      } else {
        slot = free_slots.back();
        free_slots.pop_back();
      }
      slot_of[p->yasai_save_id()] = slot;
      p->yasai_set_save_slot(slot);
    }
  }
  return slots;
}

}  // namespace yasai_contract
