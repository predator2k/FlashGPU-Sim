// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ptx_parser_INCLUDED
#define ptx_parser_INCLUDED

#include "../abstract_hardware_model.h"
#include "ptx_ir.h"

#include <initializer_list>
#include <vector>

class gpgpu_context;
typedef void *yyscan_t;
class ptx_recognizer {
 public:
  ptx_recognizer(gpgpu_context *ctx) : g_return_var(ctx) {
    scanner = NULL;
    g_size = -1;
    g_add_identifier_cached__identifier = NULL;
    g_alignment_spec = -1;
    g_var_type = NULL;
    g_opcode = -1;
    g_space_spec = undefined_space;
    g_space_spec2 = undefined_space;
    g_ptr_spec = undefined_space;
    g_scalar_type_spec = -1;
    g_vector_spec = -1;
    g_extern_spec = 0;
    g_func_decl = 0;
    g_ident_add_uid = 0;
    g_const_alloc = 1;
    g_max_regs_per_thread = 0;
    g_global_symbol_table = NULL;
    g_current_symbol_table = NULL;
    g_last_symbol = NULL;
    g_error_detected = 0;
    g_entry_func_param_index = 0;
    g_func_info = NULL;
    g_debug_ir_generation = false;
    g_vector_operand_start = -1;
    gpgpu_ctx = ctx;
  }
  // global list
  yyscan_t scanner;
#define PTX_LINEBUF_SIZE (4 * 1024)
  char linebuf[PTX_LINEBUF_SIZE];
  unsigned col;
  int g_size;
  char *g_add_identifier_cached__identifier;
  int g_add_identifier_cached__array_dim;
  int g_add_identifier_cached__array_ident;
  int g_alignment_spec;
  // variable declaration stuff:
  type_info *g_var_type;
  // instruction definition stuff:
  const symbol *g_pred;
  int g_neg_pred;
  int g_pred_mod;
  symbol *g_label;
  int g_opcode;
  std::list<operand_info> g_operands;
  std::list<int> g_options;
  std::list<int> g_wmma_options;
  std::list<int> g_mma_options;
  std::list<int> g_wgmma_options;
  std::list<int> g_scalar_type;
  // type specifier stuff:
  memory_space_t g_space_spec;
  memory_space_t g_space_spec2;  // For cp.async with dual address spaces (e.g., .shared.global)
  memory_space_t g_ptr_spec;
  int g_scalar_type_spec;
  int g_vector_spec;
  int g_extern_spec;
  int g_func_decl;
  int g_ident_add_uid;
  unsigned g_const_alloc;
  unsigned g_max_regs_per_thread;
  symbol_table *g_global_symbol_table;
  symbol_table *g_current_symbol_table;
  symbol *g_last_symbol;
  std::list<ptx_instruction *> g_instructions;
  int g_error_detected;
  unsigned g_entry_func_param_index;
  function_info *g_func_info;
  operand_info g_return_var;
  bool g_debug_ir_generation;
  int g_vector_operand_start;
  int g_entry_point;
  const struct core_config *g_shader_core_config;
  std::map<std::string, std::map<unsigned, const ptx_instruction *> >
      g_inst_lookup;
  // the program intermediate representation...
  std::map<std::string, symbol_table *> g_sym_name_to_symbol_table;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

  // member function list
  void init_directive_state();
  void init_instruction_state();
  // yasai: .shared::cta / .shared::cluster on ld, st, atom and red name the shared state space
  // (the lexer returns them as CTA_OPTION / CLUSTER_OPTION, which set no space).
  void shared_window_space();
  // yasai: records the function's .reqnctapercluster shape.
  void req_ncta_per_cluster(int x, int y, int z);
  // yasai: appends one label to the pending .branchtargets list.
  //   identifier: label name
  void add_branchtarget_label(const char *identifier);
  // yasai: ends a .branchtargets statement: the pending labels become the target list named
  // identifier in the current function, and identifier becomes a symbol so brx.idx can name it.
  //   identifier: name of the target list
  void add_branchtargets(const char *identifier);
  // yasai: labels of the .branchtargets statement being parsed.
  std::vector<std::string> g_branchtarget_labels;
  // yasai: indirect calls (PTX ISA call with a register target, .callprototype, .calltargets).
  // A prototype or target list is visible in the block of the function body that declares it and
  // in the blocks nested in it (two functions, or two blocks, may use one name); a call names one
  // after its arguments, and the name is resolved at the end of the function body, since the
  // declaration may come later. The functions of a target list are
  // resolved at the end of the module, since they may be defined later.
  //   start_callprototype: "<name>: .callprototype" begins (name: its label)
  //   add_proto_param: one parameter of the list being read (bytes: its size)
  //   proto_params_are_return: the list just read is the return value
  //   end_callprototype: the declaration ends; the list just read are the parameters
  //   add_calltarget_func: one function of a .calltargets list (name: the function)
  //   add_calltargets: the .calltargets statement ends (name: its label)
  //   set_call_tail: the current call instruction names a prototype or list (name)
  //   end_module: checks the target lists' functions after the whole module is parsed
  void start_callprototype(const char *name);
  void add_proto_param(long long bytes);
  void proto_params_are_return();
  void end_callprototype();
  void add_calltarget_func(const char *name);
  void add_calltargets(const char *name);
  void set_call_tail(const char *name);
  void end_module();
  // yasai: resolves the tails of the current function's calls (end_function); an unknown name
  // stops the parse rather than leaving the call unchecked.
  void resolve_call_tails();
  yasai_call_proto *g_proto_cur = nullptr;      // declaration being read
  std::vector<unsigned> g_proto_params;         // parameter sizes being read
  // current function's prototypes and lists: (declaring block, name) to the declaration
  std::map<std::pair<symbol_table *, std::string>, yasai_call_proto *> g_func_protos;
  std::map<std::pair<symbol_table *, std::string>, yasai_call_targets *> g_func_targets;
  std::vector<std::string> g_calltarget_funcs;  // functions of the list being read
  std::vector<std::pair<std::string, std::string>> g_module_target_funcs;  // (list, function)
  std::string g_call_tail;                      // tail name of the current call
  // yasai: marks the current instruction as bf16x2 (the type itself is stored as f16x2); no
  // effect outside an instruction (a .reg .bf16x2 declaration).
  void note_bf16x2();
  // yasai: the lexer saw a wmma shape with .shared or .shared::cta; take_wmma_shared() answers and clears it.
  void note_wmma_shared() { m_yasai_wmma_shared = true; }
  bool take_wmma_shared() {
    const bool s = m_yasai_wmma_shared;
    m_yasai_wmma_shared = false;
    return s;
  }
  bool m_yasai_wmma_shared = false;  // yasai: see note_wmma_shared()
  // yasai: parentheses open at this point of the lexer (the declaration directives switch the lexer to
  // names only at depth 0, so an entry's parameter list with .ptr .shared keeps the body's first opcode).
  int m_yasai_paren = 0;
  // yasai: marks the current instruction's packed small-float format (E4M3X2_TYPE and so on).
  void note_packed_minifloat(int kind);
  void start_function(int entry_point);
  void add_function_name(const char *fname);
  void add_directive();
  void end_function();
  void add_identifier(const char *s, int array_dim, unsigned array_ident);
  void add_function_arg();
  void add_scalar_type_spec(int type_spec);
  void add_scalar_operand(const char *identifier);
  void add_neg_pred_operand(const char *identifier);
  void add_variables();
  void set_variable_type();
  void add_opcode(int opcode);
  void add_pred(const char *identifier, int negate, int predModifier);
  void add_1vector_operand(const char *d1);
  void add_2vector_operand(const char *d1, const char *d2);
  void add_3vector_operand(const char *d1, const char *d2, const char *d3);
  void add_4vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4);
  void add_5vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4, const char *d5);
  void add_8vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4, const char *d5, const char *d6,
                           const char *d7, const char *d8);
  void add_vector_operand(const std::vector<const char *> &components);
  void begin_vector_operand();
  void end_vector_operand();
  void add_option(int option);
  void add_wmma_option(int option);
  void add_mma_option(int option);
  void add_wgmma_option(int option);
  void add_builtin_operand(int builtin, int dim_modifier);
  void add_memory_operand();
  void add_literal_int(long long value);
  // yasai: marker name of an integer literal inside a tex-style vector operand.
  char *tex_literal_name(long long value);
  // yasai: pushes a vector operand whose names include tex_literal_name markers; false when none.
  bool add_vector_with_literals(std::initializer_list<const char *> names);
  // yasai: a function name in a global initializer list (for example NCCL's ncclDevFuncTable);
  // the literal value is the function's handle from yasai_func::function_handle.
  void add_function_address_literal(const char *identifier);
  void add_literal_float(float value);
  void add_literal_double(double value);
  void add_address_operand(const char *identifier, long long offset);
  void add_address_operand2(long long offset);
  void add_label(const char *idenfiier);
  void add_vector_spec(int spec);
  void add_space_spec(enum _memory_space_t spec, int value);
  void add_ptr_spec(enum _memory_space_t spec);
  void add_extern_spec();
  void add_instruction();
  void set_return();
  void add_alignment_spec(int spec);
  void add_array_initializer();
  void add_file(unsigned num, const char *filename);
  void add_version_info(float ver, unsigned ext);
  void *reset_symtab();
  void set_symtab(void *);
  void add_pragma(const char *str);
  void func_header(const char *a);
  void func_header_info(const char *a);
  void func_header_info_int(const char *a, int b);
  void add_constptr(const char *identifier1, const char *identifier2,
                    int offset);
  void target_header(char *a);
  void target_header2(char *a, char *b);
  void target_header3(char *a, char *b, char *c);
  void add_double_operand(const char *d1, const char *d2);
  void change_memory_addr_space(const char *identifier);
  void change_operand_lohi(int lohi);
  void change_double_operand_type(int addr_type);
  void change_operand_neg();
  void set_immediate_operand_type();
  void version_header(double a);
  void maxnt_id(int x, int y, int z);
  void parse_error_impl(const char *file, unsigned line, const char *msg, ...);
  void parse_assert_impl(int test_value, const char *file, unsigned line,
                         const char *msg, ...);
  // Jin: handle instructino group for cdp
  void start_inst_group();
  void end_inst_group();
  bool check_for_duplicates(const char *identifier);
  void read_parser_environment_variables();
  void set_ptx_warp_size(const struct core_config *warp_size);
  const class ptx_instruction *ptx_instruction_lookup(const char *filename,
                                                      unsigned linenumber);
};

const char *decode_token(int type);
void read_parser_environment_variables();

#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3

#endif
