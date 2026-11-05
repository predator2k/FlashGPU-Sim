// yasai_contract.h: contraction of a mul and an add into one fused multiply-add, as ptxas does
// (yasai fakeenv simulation backend).
//
// PTX lets the code generator contract a mul and an add (or sub) that carry no rounding modifier
// into one fused multiply-add that rounds once (PTX ISA, add and mul notes). ptxas does so, and the
// GPU then computes a different result than two separately rounded instructions. The rules below
// were measured, not taken from documentation: 28 probe cases (tests/contract_cases.py, 92 to 102
// PTX modules per target, 8,192 elements each, inputs where one and two roundings differ), ptxas
// of CUDA 13.0 (V13.0.88) at -O3 for sm_90, sm_100a, sm_120 and sm_121 (SASS of the test kernels:
// FFMA, FFMA2, DFMA or HFMA2 with a real addend), and the numbers on an RTX 5090 (sm_120, driver
// 580.159.03) and a GB10 (sm_121, driver 580.95.05), where ptxas -O3 and the driver's JIT agreed in
// every case and the four targets agreed with each other. A mul M and an add or sub A are
// contracted when:
//   * both are of one type among f32, f64, f16, bf16, f16x2, bf16x2 and f32x2, M is a mul without
//     .sat, and neither carries a predicate;
//   * neither carries a rounding modifier (.rn and so on), except .rn on f32x2: an RTX 5090 and a GB10
//     contract f32x2 even with .rn on either instruction (ptxas emits FFMA on sm_12x and FFMA2 on
//     sm_100a, also with --fmad=false), unlike the PTX ISA text, and the simulator follows the
//     hardware; with .rz, .rm or .rp on either instruction f32x2 is not contracted (an RTX 5090, all
//     pairs of the modifiers, tests/contract_cases.py --directed);
//   * .ftz is on both or on neither;
//   * A and M are in one basic block (a label or a branch, call, return, exit or barrier between
//     them separates them), and every use of M's result is an add or sub of that kind in the
//     block, or a neg of that kind whose result is used only so (not for bf16x2); a product that is
//     also stored or used otherwise is not contracted, nor one used in another block;
//   * when both sources of A are such products, the one at A's first source position is fused and
//     the other is rounded first (position, not the order of definition); when the first is not
//     contractible, the second is fused.
// Uses of M's result by two adds contract both; M's source registers may be written between M and
// A (ptxas uses the values M read). add.sat contracts (FFMA.SAT). -O0 and --fmad=false do not
// contract (recorded only; no switch for them).
//
// The simulator marks the pairs when a function is parsed (analyze) and gives each pair a save slot
// once the function's final instruction order is known (assign_slots, after the PTX reordering of
// ptx_assemble). At execution, M writes the values of its two sources into its slot of the
// executing thread as it reads them, and A computes the fused result from them. A slot is busy from
// M to the last add that fuses M's product and is then reused; M's source registers may be written
// in between. YASAI_FMA_CONTRACT selects the behaviour and has no default (user rule 2026-11-03):
//   ptxas  contract as above
//   off    execute every mul and add on its own (the simulator's behaviour before)
#ifndef YASAI_CONTRACT_H
#define YASAI_CONTRACT_H

#include <list>
#include <vector>

class ptx_instruction;
class symbol_table;

namespace yasai_contract {

/// Whether contraction is on (YASAI_FMA_CONTRACT=ptxas); stops the process when the variable is
/// unset or holds another value.
bool enabled();

/// Marks the contractible (mul, add) pairs of one function's instructions (yasai_contract.h): each
/// contracted mul gets a pair id, and each add or sub that fuses its product the same id. The time
/// grows with the function's length (one pass that lists each register's readers and writers),
/// rather than with the number of muls times that length.
/// @param body The function's instructions in order (labels included), as the parser keeps them.
/// @param symtab The function's symbol table (not changed).
/// @param filename Source file of the function (not used).
/// @return The number of add or sub instructions marked.
unsigned analyze(const std::list<ptx_instruction *> &body, symbol_table *symtab,
                 const char *filename);

/// Gives each marked pair of a function a save slot in the function's final instruction order:
/// the lowest slot that is free at the mul, held until the last add that fuses its product. A mul
/// whose fusing adds were all replaced by a later rewrite gets no slot. Stops the process when a
/// fused add comes before its mul, or a label or a block end lies between them (neither can happen
/// unless the reordering moved an instruction across a dependency or a block boundary).
/// @param insts The function's instructions in their final order (labels included).
/// @param func Function name for the messages.
/// @return The number of slots the function uses.
unsigned assign_slots(const std::list<ptx_instruction *> &insts, const char *func);

}  // namespace yasai_contract

#endif  // YASAI_CONTRACT_H
