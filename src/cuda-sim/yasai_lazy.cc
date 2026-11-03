// yasai_lazy.cc: see yasai_lazy.h.
#include "yasai_lazy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <unordered_map>

#include <unistd.h>

#include "../../libcuda/gpgpu_context.h"
#include "../../libcuda/yasai_fakeenv_rec.h"
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpusim_entrypoint.h"
#include "../stream_manager.h"
#include "../yasai_realtime.h"
#include "cuda-sim.h"
#include "host_window.h"
#include "opcodes.h"
#include "ptx.tab.h"
#include "ptx_ir.h"
#include "shared_devmem.h"
#include "yasai_func.h"

namespace yasai_lazy {

namespace {

// ---- Switches -----------------------------------------------------------------------------------

/// YASAI_LAZY_REPORT=1: one line per deferral and per flush.
bool report() {
  static const bool on = [] {
    const char *v = getenv("YASAI_LAZY_REPORT");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

/// Whether the kernel's name matches YASAI_LAZY_EAGER_REGEX (the force-eager list).
bool force_eager(const std::string &name) {
  static const char *rx_s = getenv("YASAI_LAZY_EAGER_REGEX");
  if (rx_s == nullptr || *rx_s == '\0') return false;
  static const std::regex rx(rx_s);
  return std::regex_search(name, rx);
}

// ---- PTX analysis -------------------------------------------------------------------------------

/// Where a register's value may come from (a flow-insensitive over-approximation).
struct Prov {
  bool ptrload = false;                // a 64-bit value loaded from memory
  bool localsh = false;                // a shared, local or parameter-local address
  std::set<uint64_t> words;            // parameter words (byte offsets) holding the value
  std::set<const symbol *> structs;    // parameters any of whose words may hold the value
  std::set<uint64_t> paddr;            // parameter addresses: offset, or kUnknown | param offset
  std::set<const symbol *> gsyms;      // global variables whose address it is
  /// Adds o; returns whether anything changed.
  bool merge(const Prov &o) {
    bool ch = false;
    if (o.ptrload && !ptrload) ptrload = ch = true;
    if (o.localsh && !localsh) localsh = ch = true;
    for (uint64_t w : o.words) ch = words.insert(w).second || ch;
    for (const symbol *s : o.structs) ch = structs.insert(s).second || ch;
    for (uint64_t a : o.paddr) ch = paddr.insert(a).second || ch;
    for (const symbol *s : o.gsyms) ch = gsyms.insert(s).second || ch;
    return ch;
  }
  bool empty() const {
    return !ptrload && !localsh && words.empty() && structs.empty() && paddr.empty() &&
           gsyms.empty();
  }
};

/// Marks a parameter address whose offset inside the parameter is not known.
constexpr uint64_t kUnknown = 1ull << 63;

/// The accesses of one function, as parameter references.
struct FuncAccess {
  bool bounded = true;                       // every global access is covered
  std::string eager;                         // why the kernel must run eagerly ("" if none)
  std::set<uint64_t> read_words, write_words;  // parameter word offsets
  std::set<const symbol *> read_structs, write_structs;  // whole parameters
  std::set<const symbol *> read_gsyms, write_gsyms;      // global variables
};

/// Whether a PTX type is 64 bits wide.
bool is64(int t) { return t == U64_TYPE || t == S64_TYPE || t == B64_TYPE || t == F64_TYPE; }

/// The instruction's text from the opcode to the first blank (its qualifiers).
std::string head_of(const ptx_instruction *pI) {
  std::string s = pI->get_source();
  size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return s;
  if (s[b] == '@') {  // guard predicate
    b = s.find_first_of(" \t", b);
    b = b == std::string::npos ? b : s.find_first_not_of(" \t", b);
    if (b == std::string::npos) return s;
  }
  const size_t e = s.find_first_of(" \t", b);
  return s.substr(b, e == std::string::npos ? std::string::npos : e - b);
}

/// Whether a symbol is a kernel parameter.
bool is_param_sym(const symbol *s) {
  return s != nullptr && (s->is_param_kernel() ||
                          (s->type() != nullptr && s->type()->get_key().is_param_kernel()));
}

/// Calls f for every register symbol of operand op (a register or a vector of registers).
template <typename F>
void each_reg(const operand_info &op, F f) {
  if (op.is_vector()) {
    for (unsigned e = 0; e < op.get_vect_nelem(); e++)
      if (const symbol *s = op.vec_symbol_or_null(e))
        if (s->is_reg()) f(s, e);
  } else if (op.is_reg()) {
    f(op.get_symbol(), 0u);
  }
}

/// The analysis of one function body.
class Analyzer {
 public:
  explicit Analyzer(function_info *f) : m_f(f) {}

  /// Runs the analysis; the result is in result().
  void run() {
    // Fixpoint of the register provenance: every instruction adds to its destinations.
    bool changed = true;
    int pass = 0;
    while (changed && pass < 40) {
      changed = false;
      for (const ptx_instruction *pI : m_f->yasai_instructions()) changed = transfer(pI) || changed;
      pass++;
    }
    if (changed) unbound("the provenance analysis did not converge");
    for (const ptx_instruction *pI : m_f->yasai_instructions()) classify(pI);
  }
  const FuncAccess &result() const { return m_out; }

 private:
  function_info *m_f;
  std::unordered_map<const symbol *, Prov> m_prov;
  FuncAccess m_out;

  void unbound(const std::string &why) {
    if (m_out.bounded) m_out.eager = why;
    m_out.bounded = false;
  }
  void eager(const std::string &why) {
    if (m_out.eager.empty()) m_out.eager = why;
  }

  /// Provenance of a source operand (register, vector, symbol address, literal).
  Prov prov_of(const operand_info &op) {
    Prov p;
    if (op.is_vector()) {
      each_reg(op, [&](const symbol *s, unsigned) { p.merge(m_prov[s]); });
      return p;
    }
    if (op.is_reg()) return m_prov[op.get_symbol()];
    if (op.is_literal() || op.is_label() || op.is_builtin()) return p;
    const operand_type t = op.get_type();
    if (t == symbolic_t || t == address_t || t == memory_t) {
      const symbol *s = op.get_symbol();
      if (s == nullptr) return p;
      if (is_param_sym(s)) {
        p.paddr.insert(s->has_valid_address() ? uint64_t(s->get_address()) + op.get_addr_offset()
                                              : kUnknown);
      } else if (s->is_global()) {
        p.gsyms.insert(s);
      } else if (s->is_shared() || s->is_local() || s->is_param_local() || s->is_sstarr()) {
        p.localsh = true;
      } else if (s->is_const()) {
        // the address of a constant variable: no global access goes through it
      }
    }
    return p;
  }

  /// Provenance of the address of a memory operand (its base register or symbol).
  Prov addr_prov(const operand_info &op) {
    if (op.is_memory_operand()) {
      const symbol *s = op.get_symbol();
      if (s != nullptr && s->is_reg()) {
        Prov p = m_prov[s];
        // [reg + k] on a parameter address moves the offset.
        if (op.get_addr_offset() != 0 && !p.paddr.empty()) {
          std::set<uint64_t> moved;
          for (uint64_t a : p.paddr)
            moved.insert((a & kUnknown) ? a : a + uint64_t(op.get_addr_offset()));
          p.paddr = moved;
        }
        return p;
      }
      return prov_of(op);
    }
    return prov_of(op);
  }

  /// Sets dst's register(s) to include p; returns whether anything changed.
  bool def(const operand_info &dst, const Prov &p) {
    bool ch = false;
    each_reg(dst, [&](const symbol *s, unsigned) { ch = m_prov[s].merge(p) || ch; });
    return ch;
  }

  /// The value of a load from parameter memory at the addresses of a, of the given width.
  Prov param_load(const Prov &a, bool wide, unsigned vec) {
    Prov v;
    if (!wide) return v;
    for (uint64_t pa : a.paddr) {
      if (pa & kUnknown) {
        // An unknown offset: any word of any parameter may be read; mark every parameter.
        for (const symbol *s : params()) v.structs.insert(s);
        continue;
      }
      for (unsigned e = 0; e < vec; e++) v.words.insert(pa + 8 * e);
    }
    return v;
  }

  /// The kernel's parameter symbols.
  std::vector<const symbol *> params() {
    std::vector<const symbol *> out;
    for (unsigned i = 0; i < m_f->yasai_param_count(); i++)
      if (const symbol *s = m_f->yasai_param_symbol(i)) out.push_back(s);
    return out;
  }

  /// Number of elements of an instruction's vector type.
  static unsigned vec_n(const ptx_instruction *pI) {
    switch (pI->get_vector()) {
      case V2_TYPE: return 2;
      case V3_TYPE: return 3;
      case V4_TYPE: return 4;
      case V8_TYPE: return 8;
      default: return 1;
    }
  }

  /// Whether a load or store space is parameter space.
  static bool is_param_space(memory_space_t sp) {
    const auto t = sp.get_type();
    return t == param_space_kernel || t == param_space_unclassified;
  }

  /// Adds the destination provenance of one instruction.
  bool transfer(const ptx_instruction *pI) {
    const int opc = pI->get_opcode();
    const unsigned n = pI->get_num_operands();
    if (n == 0) return false;
    switch (opc) {
      case LD_OP:
      case LDU_OP: {
        if (n < 2) return false;
        const operand_info &dst = pI->operand_lookup(0);
        const operand_info &src = pI->operand_lookup(1);
        const Prov a = addr_prov(src);
        const bool wide = is64(pI->get_type());
        if (is_param_space(pI->get_space()) || (!a.paddr.empty() && !a.ptrload)) {
          // A load from the parameter window: the value of parameter words.
          Prov v = param_load(a, wide, vec_n(pI));
          // ld.param of a struct parameter through a symbol keeps the exact word.
          return def(dst, v);
        }
        Prov v;
        v.ptrload = wide;
        return def(dst, v);
      }
      case ATOM_OP: {
        Prov v;
        v.ptrload = is64(pI->get_type());
        return def(pI->operand_lookup(0), v);
      }
      case MOV_OP:
      case CVTA_OP:
      case CVT_OP: {
        if (n < 2) return false;
        Prov p = prov_of(pI->operand_lookup(1));
        if (opc == CVTA_OP) {
          const std::string h = head_of(pI);
          if (h.find(".shared") != std::string::npos || h.find(".local") != std::string::npos)
            p.localsh = true;
        }
        return def(pI->operand_lookup(0), p);
      }
      case ADD_OP:
      case SUB_OP: {
        if (n < 3) return false;
        Prov a = prov_of(pI->operand_lookup(1));
        Prov b = prov_of(pI->operand_lookup(2));
        const operand_info &bo = pI->operand_lookup(2);
        if (bo.is_literal() && !a.paddr.empty()) {
          const long long k = bo.get_literal_value().s64;
          std::set<uint64_t> moved;
          for (uint64_t x : a.paddr)
            moved.insert((x & kUnknown) ? x : uint64_t(x + (opc == ADD_OP ? k : -k)));
          a.paddr = moved;
        } else if (!b.empty() && !a.paddr.empty()) {
          std::set<uint64_t> unk;
          for (uint64_t x : a.paddr) unk.insert(kUnknown);
          a.paddr = unk;
        }
        if (!a.empty() && !b.paddr.empty()) {
          std::set<uint64_t> unk;
          for (uint64_t x : b.paddr) unk.insert(kUnknown);
          b.paddr = unk;
        }
        a.merge(b);
        return def(pI->operand_lookup(0), a);
      }
      case MUL_OP:
      case MUL24_OP:
      case SHL_OP:
      case DIV_OP:
      case REM_OP:
        // A scaled value is an offset, not a pointer.
        return false;
      case MAD_OP:
      case MAD24_OP:
        if (n < 4) return false;
        return def(pI->operand_lookup(0), prov_of(pI->operand_lookup(3)));
      case SETP_OP:
      case SET_OP:
      case ST_OP:
      case RED_OP:
      case BRA_OP:
      case BAR_OP:
      case CALL_OP:
      case RET_OP:
      case EXIT_OP:
        return false;
      default: {
        // Any other instruction: its destination may carry what its sources carry.
        Prov p;
        for (unsigned i = 1; i < n; i++) p.merge(prov_of(pI->operand_lookup(i)));
        return def(pI->operand_lookup(0), p);
      }
    }
  }

  /// Records a global access through address provenance a.
  void access(const ptx_instruction *pI, const Prov &a, bool rd, bool wr) {
    if (a.ptrload) {
      unbound(std::string("pointer loaded from memory (") + head_of(pI) + ")");
      return;
    }
    const bool any = !a.words.empty() || !a.structs.empty() || !a.gsyms.empty();
    if (!any) {
      if (a.localsh || !a.paddr.empty()) return;  // shared, local or parameter memory
      unbound(std::string("address of unknown origin (") + head_of(pI) + ")");
      return;
    }
    if (rd) {
      m_out.read_words.insert(a.words.begin(), a.words.end());
      m_out.read_structs.insert(a.structs.begin(), a.structs.end());
      m_out.read_gsyms.insert(a.gsyms.begin(), a.gsyms.end());
    }
    if (wr) {
      m_out.write_words.insert(a.words.begin(), a.words.end());
      m_out.write_structs.insert(a.structs.begin(), a.structs.end());
      m_out.write_gsyms.insert(a.gsyms.begin(), a.gsyms.end());
    }
  }

  /// Whether a space can address global memory.
  static bool global_or_generic(memory_space_t sp) {
    const auto t = sp.get_type();
    return t == global_space || t == generic_space || t == undefined_space;
  }

  /// Classifies one instruction's global accesses and its eager causes.
  void classify(const ptx_instruction *pI) {
    const int opc = pI->get_opcode();
    const unsigned n = pI->get_num_operands();
    const std::string h = head_of(pI);
    switch (opc) {
      case LD_OP:
      case LDU_OP:
        if (n >= 2 && global_or_generic(pI->get_space())) {
          const Prov a = addr_prov(pI->operand_lookup(1));
          if (!a.paddr.empty() && a.words.empty() && a.structs.empty() && a.gsyms.empty() &&
              !a.ptrload)
            break;  // parameter window
          for (const char *q : {".volatile", ".acquire", ".relaxed", ".acq_rel", ".sc", ".mmio"})
            if (h.find(q) != std::string::npos) eager("ordered or volatile load");
          access(pI, a, true, false);
        }
        break;
      case ST_OP:
        if (n >= 1 && global_or_generic(pI->get_space())) {
          if (!pI->is_weak_store()) eager("ordered or volatile store");
          access(pI, addr_prov(pI->operand_lookup(0)), false, true);
        }
        break;
      case ATOM_OP:
        // Shared-memory atomics stay inside the CTA; global (or generic) ones can make kernels
        // wait for each other.
        if (n >= 2 && global_or_generic(pI->get_space())) {
          const Prov a = addr_prov(pI->operand_lookup(1));
          if (a.localsh && a.words.empty() && a.structs.empty() && a.gsyms.empty() && !a.ptrload)
            break;
          eager("global atomic");
          access(pI, a, true, true);
        }
        break;
      case RED_OP:
        if (n >= 1 && global_or_generic(pI->get_space())) {
          const Prov a = addr_prov(pI->operand_lookup(0));
          if (a.localsh && a.words.empty() && a.structs.empty() && a.gsyms.empty() && !a.ptrload)
            break;
          eager("global reduction");
          access(pI, a, true, true);
        }
        break;
      case CP_ASYNC_OP:
        if (n >= 2) access(pI, addr_prov(pI->operand_lookup(1)), true, false);
        break;
      case TMA_OP: {
        // Every operand that carries a global pointer or a tensor map: read and written. The
        // tensor map of a .tensor copy must lie in the parameters (__grid_constant__): one in
        // global memory names its tensor through a pointer read from memory.
        const bool tensor = h.find(".tensor") != std::string::npos;
        for (unsigned i = 0; i < n; i++) {
          const operand_info &op = pI->operand_lookup(i);
          if (op.is_literal() || op.is_label()) continue;
          Prov a = addr_prov(op);
          if (a.ptrload) {
            unbound(std::string("tensor map or address loaded from memory (") + h + ")");
            continue;
          }
          if (tensor && (!a.words.empty() || !a.structs.empty() || !a.gsyms.empty())) {
            unbound(std::string("tensor map in global memory (") + h + ")");
            continue;
          }
          // A tensor map in the parameters: the pointers it holds.
          for (uint64_t pa : a.paddr) {
            for (const symbol *s : params()) {
              const uint64_t lo = s->has_valid_address() ? s->get_address() : 0;
              const uint64_t hi = lo + s->get_size_in_bytes();
              if ((pa & kUnknown) || (pa >= lo && pa < hi)) a.structs.insert(s);
            }
          }
          a.paddr.clear();
          if (a.words.empty() && a.structs.empty() && a.gsyms.empty()) continue;
          access(pI, a, true, true);
        }
        break;
      }
      case MULTIMEM_LD_REDUCE_OP:
      case MULTIMEM_ST_OP:
      case MULTIMEM_RED_OP:
        eager("multimem");
        unbound("multimem");
        break;
      case TENSORMAP_OP:
        eager("tensormap update");
        unbound("tensormap update");
        break;
      case CLC_TRY_CANCEL_OP:
        eager("cluster launch control");
        break;
      case TEX_OP:
      case SULD_OP:
      case SUST_OP:
      case SURED_OP:
        eager("texture or surface");
        unbound("texture or surface");
        break;
      case MEMBAR_OP:
        if (h.find(".cta") == std::string::npos) eager("membar above CTA scope");
        break;
      case FENCE_OP:
        if (h.find(".proxy") == std::string::npos && h.find("mbarrier_init") == std::string::npos &&
            h.find(".cta") == std::string::npos)
          eager("fence above CTA scope");
        break;
      case CALL_OP: {
        const operand_info &target = pI->func_addr();
        function_info *f = target.is_function_address() ? target.get_symbol()->get_pc() : nullptr;
        if (f == nullptr) {
          unbound("indirect call");
          break;
        }
        const std::string &name = f->get_name();
        if (name == "vprintf") {
          eager("printf");
        } else if (name == "malloc" || name == "free") {
          eager("device malloc");
        } else if (name.compare(0, 4, "cuda") == 0) {
          eager("device runtime call");
        } else if (name == "__assertfail") {
          // only stops the kernel
        } else if (f->get_function_size() == 0) {
          unbound("call of a body that is not loaded");
        } else if (callee_touches_memory(f)) {
          unbound("call of a function that accesses global memory (" + name + ")");
        }
        break;
      }
      case CALLP_OP:
      case BRX_OP:
        if (opc == CALLP_OP) unbound("callp");
        break;
      default:
        break;
    }
  }

  /// Whether f, or a function it calls, may access global memory or synchronize.
  static bool callee_touches_memory(function_info *f) {
    std::set<function_info *> seen;
    std::vector<function_info *> todo{f};
    while (!todo.empty()) {
      function_info *g = todo.back();
      todo.pop_back();
      if (!seen.insert(g).second) continue;
      if (g->get_function_size() == 0) return true;
      for (const ptx_instruction *pI : g->yasai_instructions()) {
        const int opc = pI->get_opcode();
        const memory_space_t sp = pI->get_space();
        switch (opc) {
          case LD_OP:
          case LDU_OP:
          case ST_OP:
            if (global_or_generic(sp)) return true;
            break;
          case ATOM_OP:
          case RED_OP:
          case CP_ASYNC_OP:
          case TMA_OP:
          case MULTIMEM_LD_REDUCE_OP:
          case MULTIMEM_ST_OP:
          case MULTIMEM_RED_OP:
          case TEX_OP:
          case SULD_OP:
          case SUST_OP:
          case SURED_OP:
          case CALLP_OP:
            return true;
          case CALL_OP: {
            const operand_info &t = pI->func_addr();
            function_info *c = t.is_function_address() ? t.get_symbol()->get_pc() : nullptr;
            if (c == nullptr) return true;
            const std::string &name = c->get_name();
            if (name == "vprintf" || name == "malloc" || name == "free" ||
                name.compare(0, 4, "cuda") == 0)
              return true;
            if (name != "__assertfail") todo.push_back(c);
            break;
          }
          default:
            break;
        }
      }
    }
    return false;
  }
};

/// Analyses of the functions, keyed by function (computed once, on any thread).
std::mutex &analysis_mu() {
  static auto *m = new std::mutex;
  return *m;
}
std::unordered_map<const function_info *, std::shared_ptr<FuncAccess>> &analyses() {
  static auto *m = new std::unordered_map<const function_info *, std::shared_ptr<FuncAccess>>;
  return *m;
}

/// The analysis of kernel entry f.
std::shared_ptr<FuncAccess> analysis_of(function_info *f) {
  {
    std::lock_guard<std::mutex> lk(analysis_mu());
    auto it = analyses().find(f);
    if (it != analyses().end()) return it->second;
  }
  Analyzer an(f);
  an.run();
  auto r = std::make_shared<FuncAccess>(an.result());
  std::lock_guard<std::mutex> lk(analysis_mu());
  return analyses().emplace(f, r).first->second;
}

// ---- Launch sets ---------------------------------------------------------------------------------

/// Device heap bounds of the simulator (addresses that are device memory of this process).
bool in_device_heap(gpgpu_t *gpu, uint64_t v) {
  return v >= GLOBAL_HEAP_START && v < gpu->dev_malloc_top();
}

/// Adds the range that pointer value v may reach to out; sets info->reason and returns false
/// when the kernel must run eagerly. from_struct: v is any word of a parameter structure (a
/// value outside device memory is plain data then).
bool add_pointer(gpgpu_t *gpu, uint64_t v, bool from_struct, std::vector<Range> &out,
                 LaunchInfo &info) {
  if (v == 0) return true;
  host_window::range_info hr;
  if (host_window::lookup(v, 1, &hr) && hr.kind != host_window::range_kind::device) {
    info.reason = "host-visible memory";
    return false;
  }
  uint64_t base = 0;
  size_t size = 0;
  bool shared = false;
  if (shared_devmem::private_range_of(v, &base, &size, &shared)) {
    if (shared) {
      info.reason = "memory shared with another process";
      return false;
    }
    out.push_back(Range{base, base + size});
    return true;
  }
  if (in_device_heap(gpu, v)) {
    info.reason = "pointer into an allocation the registry does not know";
    return false;
  }
  if (from_struct) return true;  // not a device pointer: plain data in a parameter
  info.reason = "pointer parameter outside device memory";
  return false;
}

/// Adds the ranges of a parameter word, a whole parameter or a global variable.
bool add_word(gpgpu_t *gpu, kernel_info_t &k, uint64_t off, std::vector<Range> &out,
              LaunchInfo &info) {
  uint64_t v = 0;
  k.get_param_memory()->read(off, 8, &v);
  return add_pointer(gpu, v, false, out, info);
}
bool add_struct(gpgpu_t *gpu, kernel_info_t &k, const symbol *s, std::vector<Range> &out,
                LaunchInfo &info) {
  if (!s->has_valid_address()) {
    info.reason = "parameter without an address";
    return false;
  }
  const uint64_t lo = s->get_address();
  const uint64_t hi = lo + s->get_size_in_bytes();
  for (uint64_t o = (lo + 7) & ~7ull; o + 8 <= hi; o += 8) {
    uint64_t v = 0;
    k.get_param_memory()->read(o, 8, &v);
    if (!add_pointer(gpu, v, true, out, info)) return false;
  }
  return true;
}
bool add_gsym(const symbol *s, std::vector<Range> &out, LaunchInfo &info) {
  if (!s->has_valid_address()) {
    info.reason = "global variable without an address";
    return false;
  }
  const uint64_t a = s->get_address();
  host_window::range_info hr;
  if (host_window::lookup(a, 1, &hr)) {
    info.reason = "host-visible global variable";
    return false;
  }
  out.push_back(Range{a, a + (s->get_size_in_bytes() ? s->get_size_in_bytes() : 1)});
  return true;
}

/// Sorts and merges a range list.
void normalize(std::vector<Range> &v) {
  std::sort(v.begin(), v.end(), [](const Range &a, const Range &b) { return a.lo < b.lo; });
  std::vector<Range> out;
  for (const Range &r : v) {
    if (!out.empty() && r.lo <= out.back().hi) {
      out.back().hi = std::max(out.back().hi, r.hi);
    } else {
      out.push_back(r);
    }
  }
  v.swap(out);
}

// ---- Pending kernels (simulation thread) ---------------------------------------------------------

/// A merged set of ranges with overlap queries.
class RangeSet {
 public:
  void add(const Range &r) {
    if (r.hi <= r.lo) return;
    uint64_t lo = r.lo, hi = r.hi;
    auto it = m.upper_bound(lo);
    if (it != m.begin()) {
      auto p = std::prev(it);
      if (p->second >= lo) {
        lo = p->first;
        hi = std::max(hi, p->second);
        it = m.erase(p);
      }
    }
    while (it != m.end() && it->first <= hi) {
      hi = std::max(hi, it->second);
      it = m.erase(it);
    }
    m[lo] = hi;
  }
  void add(const std::vector<Range> &v) {
    for (const Range &r : v) add(r);
  }
  bool overlaps(const Range &r) const {
    auto it = m.upper_bound(r.lo);
    if (it != m.begin() && std::prev(it)->second > r.lo) return true;
    return it != m.end() && it->first < r.hi;
  }
  bool overlaps(const std::vector<Range> &v) const {
    for (const Range &r : v)
      if (overlaps(r)) return true;
    return false;
  }
  bool empty() const { return m.empty(); }

 private:
  std::map<uint64_t, uint64_t> m;  // first address -> end, disjoint
};

/// One pending kernel.
struct Pending {
  kernel_info_t *k;                  // the launch (owned)
  std::shared_ptr<LaunchInfo> info;  // its sets
  uint64_t seq;                      // deferral order
};

/// Module state.
struct State {
  std::mutex mu;                      // guards requests (pending is the simulation thread's)
  std::condition_variable cv;         // signalled when a request is served
  std::deque<Pending> pending;        // in launch order
  uint64_t seq = 0;                   // deferrals so far
  cuda_sim *sim = nullptr;            // the functional simulator
  // Host-thread requests: reads, writes, all; served counter.
  struct Request {
    std::vector<Range> reads, writes;
    bool all = false;
    const char *why = "";
    bool done = false;
  };
  std::vector<Request *> requests;
  std::atomic<bool> has_requests{false};
  // Statistics.
  std::atomic<uint64_t> launches{0};       // launches prepared
  std::atomic<uint64_t> deferred{0};       // launches deferred
  std::atomic<uint64_t> executed_later{0}; // deferred launches run by a flush
  std::atomic<uint64_t> flushes{0};        // observations that ran a pending kernel
  double exec_s = 0;                       // wall seconds of deferred kernels run later
  std::map<std::string, uint64_t> eager_reasons;  // eager launches by cause (guarded by mu)
};

State &st() {
  static auto *s = new State;  // never destroyed: used during exit
  return *s;
}

thread_local bool t_sim_thread = false;  // see set_sim_thread()
thread_local int t_flush_depth = 0;      // flushes running on this thread (no nesting)

/// Runs pending kernel p to completion on the simulation thread.
void run_pending(Pending &p) {
  State &s = st();
  kernel_info_t &k = *p.k;
  if (report())
    printf("yasai-lazy: running deferred kernel %s uid=%u\n", k.name().c_str(), k.get_uid());
  yasai_rec::Recorder &rec = yasai_rec::Recorder::get();
  if (rec.on() && p.info->corr != 0)
    rec.write("\"rec\":\"deferred_exec\",\"corr\":%u,\"t\":%llu", p.info->corr,
              (unsigned long long)yasai_rec::mono_ns());
  const double t0 = yasai_real_mono_s();
  k.set_next_cta_id(dim3(0, 0, 0));
  // The flush may run inside another run's turn (a copy in its stream, an eager kernel's
  // start): that run's kernel scope is restored afterwards.
  yasai_func::KernelScope *scope = yasai_func::scope_get();
  // The flush can run while the GPU model is not in the functional mode (no functional kernel
  // queued: a copy or a host request between kernels); the instructions that ask for the warp
  // id (shfl, match, vote) need it.
  gpgpu_sim *gpu = s.sim->gpgpu_ctx->the_gpgpusim->g_the_gpu;
  const bool was_functional = gpu->is_functional_sim();
  gpu->yasai_begin_functional();
  gpu->set_cache_config(k.name());  // as at the kernel's turn (stream_operation::do_operation)
  {
    yasai_func_run run(s.sim, k, false, true);
    while (!run.step()) {
    }
  }
  if (!was_functional) gpu->yasai_end_functional();
  yasai_func::scope_set(scope);
  s.exec_s += yasai_real_mono_s() - t0;
  s.executed_later++;
  k.notify_parent_finished();
  delete p.k;
  p.k = nullptr;
}

/// Runs the pending kernels that an access reading obs_r and writing obs_w depends on (all of
/// them when all is set), in launch order. Simulation thread only.
void flush_locked_free(const std::vector<Range> &obs_r, const std::vector<Range> &obs_w, bool all,
                       const char *why) {
  State &s = st();
  if (s.pending.empty()) return;
  if (t_flush_depth > 0) return;  // a kernel run by a flush makes no observation of its own
  std::vector<char> sel(s.pending.size(), 0);
  RangeSet need_r, need_w;  // what later selected work reads and writes
  need_r.add(obs_r);
  need_w.add(obs_w);
  size_t nsel = 0;
  for (size_t i = s.pending.size(); i-- > 0;) {
    const LaunchInfo &li = *s.pending[i].info;
    bool take = all || need_r.overlaps(li.writes) || need_w.overlaps(li.writes) ||
                need_w.overlaps(li.reads);
    if (!take) continue;
    sel[i] = 1;
    nsel++;
    need_r.add(li.reads);
    need_w.add(li.writes);
  }
  if (nsel == 0) return;
  s.flushes++;
  if (report())
    printf("yasai-lazy: %s runs %zu of %zu pending kernels\n", why, nsel, s.pending.size());
  std::deque<Pending> keep, run;
  for (size_t i = 0; i < s.pending.size(); i++) (sel[i] ? run : keep).push_back(s.pending[i]);
  s.pending.swap(keep);
  t_flush_depth++;
  for (Pending &p : run) run_pending(p);
  t_flush_depth--;
}

/// Whether every range of a deferred launch is still private device memory (no segment, not
/// host-visible).
bool still_private(const LaunchInfo &info) {
  for (const std::vector<Range> *v : {&info.reads, &info.writes})
    for (const Range &r : *v) {
      host_window::range_info hr;
      if (host_window::lookup(r.lo, 1, &hr) && hr.kind != host_window::range_kind::device)
        return false;
      uint64_t base = 0;
      size_t size = 0;
      bool shared = false;
      if (shared_devmem::private_range_of(r.lo, &base, &size, &shared) && shared) return false;
    }
  return true;
}

/// Process id of the simulation thread's process (0 before it started).
std::atomic<int> g_sim_pid{0};

/// Asks the simulation thread to flush and waits (host threads). Returns without waiting when
/// no simulation thread serves this process (not started, or a forked child before its restart),
/// and after YASAI_LAZY_WAIT_S seconds (default 600) with a warning, so that an exit whose
/// simulation thread has stopped does not hang (the pending kernels are then never run).
void request(const std::vector<Range> &r, const std::vector<Range> &w, bool all, const char *why) {
  if (g_sim_pid.load(std::memory_order_acquire) != getpid()) return;
  static const int wait_s = [] {
    const char *v = getenv("YASAI_LAZY_WAIT_S");
    return (v != nullptr && *v != '\0') ? atoi(v) : 600;
  }();
  State &s = st();
  State::Request req;
  req.reads = r;
  req.writes = w;
  req.all = all;
  req.why = why;
  std::unique_lock<std::mutex> lk(s.mu);
  s.requests.push_back(&req);
  s.has_requests.store(true, std::memory_order_release);
  if (!s.cv.wait_for(lk, std::chrono::seconds(wait_s), [&] { return req.done; })) {
    // Withdraw the request (the simulation thread does not serve it any more).
    for (auto it = s.requests.begin(); it != s.requests.end(); ++it)
      if (*it == &req) {
        s.requests.erase(it);
        break;
      }
    if (s.requests.empty()) s.has_requests.store(false, std::memory_order_release);
    fprintf(stderr, "yasai-lazy: WARNING: %s waited %d s for the simulation thread; the pending "
            "kernels it depends on did not run\n", why, wait_s);
  }
}

/// Whether the simulation thread has pending kernels (read without the lock by host threads: a
/// stale "none" only means the host access does not need a flush, as no kernel launched before
/// the access is pending; deferrals happen in stream order before the access's own turn).
std::atomic<size_t> g_pending_count{0};

}  // namespace

bool enabled() {
  static const bool on = [] {
    const char *v = getenv("YASAI_LAZY_EXEC");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

void set_sim_thread() {
  t_sim_thread = true;
  g_sim_pid.store(getpid(), std::memory_order_release);
}

std::shared_ptr<LaunchInfo> prepare(kernel_info_t &k, unsigned corr, bool host_fill) {
  auto info = std::make_shared<LaunchInfo>();
  info->corr = corr;
  k.set_yasai_lazy(info);
  if (!enabled()) {
    info->reason = "off";
    return info;
  }
  // The summary line at exit (a forked child inherits the registration and prints its own).
  static std::once_flag summary_once;
  std::call_once(summary_once, [] { atexit([] { print_summary(); }); });
  st().launches++;
  gpgpu_t *gpu = k.entry()->gpgpu_ctx->the_gpgpusim->g_the_gpu;
  function_info *f = k.entry();
  const std::shared_ptr<FuncAccess> fa = analysis_of(f);
  info->bounded = fa->bounded;
  if (!fa->bounded) {
    info->reason = fa->eager;
  } else {
    bool ok = true;
    for (uint64_t o : fa->read_words) ok = ok && add_word(gpu, k, o, info->reads, *info);
    for (uint64_t o : fa->write_words) ok = ok && add_word(gpu, k, o, info->writes, *info);
    for (const symbol *s : fa->read_structs) ok = ok && add_struct(gpu, k, s, info->reads, *info);
    for (const symbol *s : fa->write_structs) ok = ok && add_struct(gpu, k, s, info->writes, *info);
    for (const symbol *s : fa->read_gsyms) ok = ok && add_gsym(s, info->reads, *info);
    for (const symbol *s : fa->write_gsyms) ok = ok && add_gsym(s, info->writes, *info);
    if (!ok) {
      // The sets cannot be bounded, or they reach host-visible memory: the kernel runs
      // eagerly and flushes every pending kernel first.
      info->bounded = false;
      info->reads.clear();
      info->writes.clear();
    } else {
      normalize(info->reads);
      normalize(info->writes);
      if (!fa->eager.empty()) info->reason = fa->eager;
    }
  }
  if (info->reason.empty()) {
    if (!yasai_func::config().trace_dir.empty()) info->reason = "trace";  // launch-order traces
    else if (host_fill) info->reason = "host fill";
    else if (k.yasai_cooperative()) info->reason = "cooperative";
    else if (k.cluster_size() > 1) info->reason = "cluster";
    else if (yasai_func_interleaves(k.name())) info->reason = "CTA interleaving";
    else if (force_eager(k.name())) info->reason = "YASAI_LAZY_EAGER_REGEX";
  }
  info->deferred = info->reason.empty();
  if (info->deferred) {
    st().deferred++;
  } else {
    std::lock_guard<std::mutex> lk(st().mu);
    st().eager_reasons[info->reason]++;
  }
  if (report())
    printf("yasai-lazy: launch %s %s%s%s\n", k.name().c_str(), info->deferred ? "deferred" : "eager",
           info->deferred ? "" : ": ", info->reason.c_str());
  return info;
}

bool start(kernel_info_t &k, cuda_sim *sim) {
  State &s = st();
  s.sim = sim;
  if (!enabled()) return false;
  std::shared_ptr<LaunchInfo> info = k.yasai_lazy();
  if (!info) {
    // A launch that did not pass prepare() (a graph replay, a device-side launch): eager, with
    // the sets computed now.
    info = prepare(k, 0, k.yasai_host_fill());
    info->deferred = false;
  }
  if (info->started) return false;
  info->started = true;
  if (info->deferred && !still_private(*info)) {
    // The memory became shared (an export or a binding) between the launch and its turn: the
    // kernel runs now. Its record says exec_deferred, so its deferred_exec record follows.
    info->deferred = false;
    info->reason = "memory shared after the launch";
    yasai_rec::Recorder &rec = yasai_rec::Recorder::get();
    if (rec.on() && info->corr != 0)
      rec.write("\"rec\":\"deferred_exec\",\"corr\":%u,\"t\":%llu", info->corr,
                (unsigned long long)yasai_rec::mono_ns());
    s.executed_later++;
  }
  if (info->deferred) {
    // At most YASAI_LAZY_MAX_PENDING (default 20000) kernels stay pending, since each keeps its
    // launch state (parameter memory): beyond it the oldest one runs (it depends on no other
    // pending kernel).
    static const size_t max_pending = [] {
      const char *v = getenv("YASAI_LAZY_MAX_PENDING");
      const long n = (v != nullptr && *v != '\0') ? atol(v) : 20000;
      return size_t(n < 1 ? 1 : n);
    }();
    if (s.pending.size() >= max_pending && t_flush_depth == 0) {
      Pending oldest = s.pending.front();
      s.pending.pop_front();
      s.flushes++;
      t_flush_depth++;
      run_pending(oldest);
      t_flush_depth--;
    }
    s.pending.push_back(Pending{&k, info, s.seq++});
    g_pending_count.store(s.pending.size(), std::memory_order_release);
    if (report()) printf("yasai-lazy: kernel %s uid=%u pending\n", k.name().c_str(), k.get_uid());
    return true;
  }
  if (info->bounded)
    flush_locked_free(info->reads, info->writes, false, "eager kernel");
  else
    flush_locked_free({}, {}, true, "unbounded eager kernel");
  g_pending_count.store(s.pending.size(), std::memory_order_release);
  // A kernel run by the flush set its own cache configuration; this one's applies again.
  k.entry()->gpgpu_ctx->the_gpgpusim->g_the_gpu->set_cache_config(k.name());
  return false;
}

void observe(uint64_t r, uint64_t rn, uint64_t w, uint64_t wn) {
  if (!enabled() || g_pending_count.load(std::memory_order_acquire) == 0) return;
  std::vector<Range> rr, ww;
  if (rn) rr.push_back(Range{r, r + rn});
  if (wn) ww.push_back(Range{w, w + wn});
  if (t_sim_thread) {
    flush_locked_free(rr, ww, false, "copy or memset");
    g_pending_count.store(st().pending.size(), std::memory_order_release);
  } else {
    request(rr, ww, false, "host access");
  }
}

void observe_all(const char *why) {
  if (!enabled() || g_pending_count.load(std::memory_order_acquire) == 0) return;
  if (t_sim_thread) {
    flush_locked_free({}, {}, true, why);
    g_pending_count.store(st().pending.size(), std::memory_order_release);
  } else {
    request({}, {}, true, why);
  }
}

bool requests_waiting() { return st().has_requests.load(std::memory_order_acquire); }

void serve_requests() {
  State &s = st();
  std::vector<State::Request *> reqs;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    reqs.swap(s.requests);
    s.has_requests.store(false, std::memory_order_release);
  }
  if (reqs.empty()) return;
  for (State::Request *q : reqs) flush_locked_free(q->reads, q->writes, q->all, q->why);
  g_pending_count.store(s.pending.size(), std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(s.mu);
    for (State::Request *q : reqs) q->done = true;
  }
  s.cv.notify_all();
}

void print_summary() {
  if (!enabled()) return;
  State &s = st();
  std::string reasons;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    for (const auto &kv : s.eager_reasons)
      reasons += " [" + kv.first + "]=" + std::to_string(kv.second);
  }
  const uint64_t def = s.deferred.load(), later = s.executed_later.load();
  printf("yasai-lazy: summary launches=%llu deferred=%llu executed_later=%llu never_executed=%llu "
         "flushes=%llu deferred_exec_s=%.1f eager:%s\n",
         (unsigned long long)s.launches.load(), (unsigned long long)def,
         (unsigned long long)later, (unsigned long long)(def - later),
         (unsigned long long)s.flushes.load(), s.exec_s, reasons.c_str());
  fflush(stdout);
}

}  // namespace yasai_lazy
