// yasai_contract.cc: see yasai_contract.h.
#include "yasai_contract.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

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

/// Whether operand op names register r (as a register, a vector element or a memory base).
bool names(const operand_info &op, const symbol *r) {
  if (!op.is_valid()) return false;
  if (op.is_vector()) {
    for (unsigned i = 0; i < op.get_vect_nelem(); i++)
      if (op.vec_symbol_or_null(i) == r) return true;
    return false;
  }
  if (op.is_literal()) return false;
  switch (op.get_type()) {
    case symbolic_t: case reg_t: case memory_t: case address_t:
      return op.get_symbol() == r;
    default:
      return false;
  }
}

/// Whether the instruction writes register r through its first operand (a destination).
bool defines(const ptx_instruction *p, const symbol *r) {
  if (p->get_num_operands() == 0) return false;
  switch (p->get_opcode()) {
    case ST_OP: case RED_OP: case BRA_OP: case BRX_OP: case CALL_OP: case CALLP_OP:
      return false;
    default:
      break;
  }
  const operand_info &d = p->operand_lookup(0);
  return !d.is_memory_operand() && names(d, r);
}

/// Whether the instruction reads register r (any operand other than a destination).
bool reads(const ptx_instruction *p, const symbol *r) {
  const unsigned n = p->get_num_operands();
  for (unsigned i = 0; i < n; i++) {
    if (i == 0 && defines(p, r)) continue;
    if (names(p->operand_lookup(i), r)) return true;
  }
  if (p->has_pred() && names(p->get_pred(), r)) return true;
  return false;
}

/// Whether two instructions agree in type, bf16x2 flag and .ftz, as a contracted pair must.
bool same_kind(const ptx_instruction *m, const ptx_instruction *a) {
  return m->get_type() == a->get_type() && m->is_bf16x2() == a->is_bf16x2() &&
         m->is_ftz() == a->is_ftz();
}

/// Whether the rounding modifiers allow contraction (none, or f32x2 always).
bool rounding_ok(const ptx_instruction *p) {
  return !p->has_rounding_modifier() || p->get_type() == F32X2_TYPE;
}

/// Whether p is an add or sub that may take a contracted product of mul m.
bool fusable_add(const ptx_instruction *p, const ptx_instruction *m) {
  return (p->get_opcode() == ADD_OP || p->get_opcode() == SUB_OP) && !p->has_pred() &&
         same_kind(m, p) && rounding_ok(p) && p->get_num_operands() == 3;
}

/// The instructions of [from, to) that read r, or false when r is read elsewhere in the function
/// (outside the window) or by an instruction that is not acceptable to ok.
template <class Ok>
bool uses_in_window(const std::vector<ptx_instruction *> &insts, size_t def, size_t to,
                    const symbol *r, Ok ok, std::vector<size_t> *uses) {
  for (size_t j = 0; j < insts.size(); j++) {
    if (j == def) continue;
    const ptx_instruction *p = insts[j];
    if (p->is_label()) continue;
    if (!reads(p, r)) continue;
    if (j <= def || j >= to) return false;  // a use outside the block window
    if (!ok(j, p)) return false;
    uses->push_back(j);
  }
  return !uses->empty();
}

}  // namespace

unsigned analyze(const std::list<ptx_instruction *> &body, symbol_table *symtab,
                 const char *filename) {
  if (!enabled()) return 0;
  const std::vector<ptx_instruction *> insts(body.begin(), body.end());
  unsigned marked = 0, hidden = 0;
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
    size_t to = i + 1;
    while (to < insts.size() && !insts[to]->is_label()) {
      const ptx_instruction *p = insts[to];
      if (ends_block(p)) break;
      if (defines(p, r)) {
        to++;  // it may read r as well; later instructions see the new value
        break;
      }
      to++;
    }
    std::vector<size_t> uses;
    bool ok = uses_in_window(insts, i, to, r, [&](size_t, const ptx_instruction *p) {
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
      if (!uses_in_window(insts, u, to, nd.get_symbol(),
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
  std::map<size_t, std::pair<const symbol *, const symbol *>> saves;  // mul -> hidden registers
  for (const auto &c : cand) {
    ptx_instruction *a = insts[c.first];
    const auto it = c.second.count(1) ? c.second.find(1) : c.second.find(2);
    const prod &pr = it->second;
    ptx_instruction *m = insts[pr.mul];
    auto s = saves.find(pr.mul);
    if (s == saves.end()) {
      const symbol *md = m->dst().get_symbol();
      char na[64], nb[64];
      snprintf(na, sizeof na, "%%yasai_fma_a%u", hidden);
      snprintf(nb, sizeof nb, "%%yasai_fma_b%u", hidden);
      hidden++;
      const symbol *ha = symtab->add_variable(na, md->type(), md->get_size_in_bytes(), filename,
                                              m->source_line());
      const symbol *hb = symtab->add_variable(nb, md->type(), md->get_size_in_bytes(), filename,
                                              m->source_line());
      m->yasai_set_save(ha, hb);
      s = saves.emplace(pr.mul, std::make_pair(ha, hb)).first;
    }
    a->yasai_set_fuse(it->first, pr.neg, s->second.first, s->second.second);
    marked++;
  }
  return marked;
}

}  // namespace yasai_contract
