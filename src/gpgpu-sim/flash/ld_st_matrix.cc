#include "ld_st_matrix.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

// Implementation documentation: ld_st_matrix.md

// Direction enum for load vs store
enum class MatrixDirection { LOAD, STORE };

// Shape specification for different ldmatrix/stmatrix variants
struct LdMatrixShapeSpec {
  int rows;          // M dimension (matrix height)
  int cols;          // N dimension (matrix width)
  int lanes_per_row; // Threads per row
  int cols_per_lane; // Elements per thread (column-wise)
};

// Get shape specification based on matrix shape option
static LdMatrixShapeSpec get_shape_spec(int matrix_shape) {
  switch (matrix_shape) {
  case M8N8_OPTION:
    return {8, 8, 4, 2}; // 8 rows, 8 cols, 4 lanes/row, 2 cols/lane
  default:
    printf("Error: Unsupported matrix shape option: %d\n", matrix_shape);
    abort();
  }
}

// Common implementation for ldmatrix and stmatrix
// yasai: ldmatrix.sync.aligned.m8n16.{x1,x2,x4}.shared.b8x16.b4x16_p64 d, [a]. Every source row holds 16
// 4-bit elements packed in 8 bytes (low nibble first) followed by 8 bytes of padding; the row of
// matrix i is at the address in lane 8 * i + r. Each element is unpacked into the low nibble of a byte
// (high nibble zero), and lane 4 * r + c receives bytes 4c to 4c + 3 of row r of every matrix, one
// register per matrix. Measured on a GB10 (tests/gaps_m1.cu, cases ldm_b4x1/x2/x4). The .b6x16_p32
// form and the m16n16 shapes are not implemented and stop.
static bool yasai_ldmatrix_b4x16(const ptx_instruction *pI, core_t *core,
                                 warp_inst_t &inst) {
  bool b4 = false, b6 = false, m8n16 = false;
  int num = 0;
  for (auto option : pI->get_options()) {
    if (option == B4X16_P64_OPTION) b4 = true;
    else if (option == B6X16_P32_OPTION) b6 = true;
    else if (option == M8N16_OPTION) m8n16 = true;
    else if (option == X1_OPTION) num = 1;
    else if (option == X2_OPTION) num = 2;
    else if (option == X4_OPTION) num = 4;
  }
  if (!b4 && !b6 && !m8n16) return false;
  if (b6 || !b4 || !m8n16 || num == 0 || pI->dst().get_vect_nelem() != (unsigned)num) {
    fprintf(stderr, "yasai-ldmatrix: %s is not implemented (only m8n16.x1/x2/x4 .b8x16.b4x16_p64)\n",
            pI->to_string().c_str());
    fflush(stderr);
    abort();
  }
  inst.space.set_type(shared_space);
  const unsigned tid0 = (core->get_gpu()->is_functional_sim() ? inst.warp_id_func()
                                                              : inst.warp_id()) *
                        core->get_warp_size();
  const operand_info &addr_op = pI->src1();
  uint32_t row_addr[32];  // every lane's address first, so that d may name the address register
  for (unsigned lane = 0; lane < 32; lane++) {
    ptx_thread_info *t = core->get_thread_info()[tid0 + lane];
    row_addr[lane] = t->get_operand_value(addr_op, addr_op, U32_TYPE, t, 0).u32;
  }
  ptx_reg_t regs[4][32];
  for (unsigned lane = 0; lane < 32; lane++) {
    ptx_thread_info *t = core->get_thread_info()[tid0 + lane];
    const unsigned r = lane / 4, c = lane % 4;
    for (int i = 0; i < num; i++) {
      unsigned char packed[2];
      t->m_shared_mem->read(row_addr[8 * i + r] + 2 * c, 2, reinterpret_cast<char *>(packed));
      regs[i][lane].u64 = 0;
      for (int k = 0; k < 4; k++) {
        const unsigned nib = (packed[k / 2] >> (4 * (k % 2))) & 0xf;
        regs[i][lane].u32 |= nib << (8 * k);
      }
    }
  }
  for (unsigned lane = 0; lane < 32; lane++) {
    ptx_thread_info *t = core->get_thread_info()[tid0 + lane];
    t->set_vector_operand_values(pI->dst(), regs[0][lane], regs[1][lane], regs[2][lane],
                                 regs[3][lane]);
  }
  return true;
}

template <MatrixDirection Direction>
static void handle_ld_st_matrix_inst_impl(const ptx_instruction *pI,
                                          core_t *core, warp_inst_t &inst) {
  constexpr bool is_load = (Direction == MatrixDirection::LOAD);
  const char *inst_name = is_load ? "ldmatrix" : "stmatrix";
  if constexpr (is_load) {
    if (yasai_ldmatrix_b4x16(pI, core, inst)) return;  // yasai
  }

  // Matrix load/store encodes shared memory as an opcode modifier rather than
  // through the generic memory operand. Ensure the dynamic instruction enters
  // the shared-memory LD/ST timing path.
  inst.space.set_type(shared_space);

  // We need the thread for DPRINTF.
  int tid_lane0 = (core->get_gpu()->is_functional_sim() ? inst.warp_id_func()
                                                        : inst.warp_id()) *
                  core->get_warp_size();
  auto thread = core->get_thread_info()[tid_lane0];

  GPPRINTF_INST_EXEC(WIP, "Handling %s\n", pI->to_string().c_str());

  // Process the options.
  bool is_sync = false;
  bool is_aligned = false;

  constexpr int invalid_matrix_shape = -1;
  int matrix_shape = invalid_matrix_shape;

  constexpr int invalid_num_matrixs = -1;
  constexpr int max_num_matrixs = 4;
  int num_matrixs = invalid_num_matrixs;

  bool is_transpose = false;

  // Removed hard-coded m8n8 constants - now using shape-spec instead

  constexpr int invalid_scalar_type = -1;
  int scalar_type = invalid_scalar_type;

  for (auto option : pI->get_options()) {
    switch (option) {
    case SYNC_OPTION:
      is_sync = true;
      break;
    case ALIGNED_OPTION:
      is_aligned = true;
      break;
    case M8N8_OPTION:
      matrix_shape = option;
      break;
    case TRANS_OPTION:
      is_transpose = true;
      break;
    case X1_OPTION:
      num_matrixs = 1;
      break;
    case X2_OPTION:
      num_matrixs = 2;
      break;
    case X4_OPTION:
      num_matrixs = 4;
      break;
    case CTA_OPTION:
      // yasai: .shared::cta (ThunderKittens' stmatrix) names the CTA's shared memory, the
      // space that .shared already means.
      break;
    case B8_TYPE:
    case B16_TYPE:
      // Scalar type is consumed below from pI->get_scalar_type(). Depending on
      // the PTX grammar path, the same type token can also appear in options.
      break;
    default:
      fprintf(stderr, "%s: unknown option %d in %s\n", inst_name, option,
              pI->to_string().c_str());
      fflush(stderr);
      abort();
    }
  }

  for (auto scalar_type_option : pI->get_scalar_type()) {
    scalar_type = scalar_type_option;
  }

  /**
   * Sanity check the options. Be strict to avoid future bug.
   */
  assert(matrix_shape == M8N8_OPTION && "Matrix shape must be m8n8");
  assert(num_matrixs != invalid_num_matrixs &&
         "Number of matrixs option is required");
  assert((scalar_type == B16_TYPE || scalar_type == B8_TYPE) &&
         "Scalar type must be b16 or b8");
  assert(is_sync && "ld/stmatrix is always sync");
  assert(is_aligned && "ld/stmatrix is always aligned");
  assert(pI->get_num_operands() == 2 && "ld/stmatrix should have 2 operands");

  // Get shape specification
  LdMatrixShapeSpec spec = get_shape_spec(matrix_shape);

  // Element size in bytes
  int element_size = (scalar_type == B16_TYPE) ? 2 : 1;

  // For ldmatrix: dst is vector register, src1 is address
  // For stmatrix: dst is address, src1 is vector register
  if constexpr (is_load) {
    assert(pI->dst().get_vect_nelem() == (unsigned)num_matrixs &&
           "Destination operand size mismatch in ldmatrix");
  } else {
    assert(pI->src1().get_vect_nelem() == (unsigned)num_matrixs &&
           "Source operand size mismatch in stmatrix");
  }

  GPPRINTF_INST_EXEC(WIP,
                     "%s options: matrix_shape=%d, num_matrixs=%d, "
                     "scalar_type=%d, num_operands=%d\n",
                     inst_name, matrix_shape, num_matrixs, scalar_type,
                     pI->get_num_operands());

  auto get_u32_value = [](ptx_thread_info *thread, const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };

  // Get the address operand reference based on direction
  const auto &addr_operand = is_load ? pI->src1() : pI->dst();

  ptx_reg_t result_regs[max_num_matrixs];
  for (unsigned lane_id = 0; lane_id < core->get_warp_size(); lane_id++) {
    auto thread = core->get_thread_info()[tid_lane0 + lane_id];

    // Map lane ID to matrix position using shape-spec
    auto matrix_row_id = lane_id / spec.lanes_per_row;
    auto matrix_col_id = (lane_id % spec.lanes_per_row) * spec.cols_per_lane;

    // For stmatrix: read data from source registers first
    if constexpr (!is_load) {
      thread->get_vector_operand_values(pI->src1(), &result_regs[0],
                                        num_matrixs);
    }

    for (auto matrix_id = 0; matrix_id < num_matrixs; matrix_id++) {

      auto &data = result_regs[matrix_id];

      if (!is_transpose) {
        // Non-transpose: load contiguous elements from same row
        // Thread provides row address, accesses consecutive columns
        auto address_lane_id = matrix_id * spec.rows + matrix_row_id;
        auto address_thread =
            core->get_thread_info()[tid_lane0 + address_lane_id];
        auto base_address = get_u32_value(address_thread, addr_operand);
        uint32_t address = base_address + matrix_col_id * element_size;

        // Number of bytes to transfer: cols_per_lane * element_size
        int transfer_size = spec.cols_per_lane * element_size;

        if constexpr (is_load) {
          thread->m_shared_mem->read(address, transfer_size,
                                     reinterpret_cast<char *>(data.u16_2));
        } else {
          thread->m_shared_mem->write(address, transfer_size,
                                      reinterpret_cast<char *>(data.u16_2),
                                      thread, pI);
          // yasai: YASAI_WGMMA_DEBUG=3 prints the first 64 stmatrix stores
          // (lane 0 of each).
          if (getenv("YASAI_WGMMA_DEBUG") &&
              atoi(getenv("YASAI_WGMMA_DEBUG")) >= 3 && lane_id == 0) {
            static int printed = 0;
            if (printed++ < 64)
              printf("yasai-stmatrix tid0=%u matrix=%d addr=0x%x data=0x%08x\n",
                     tid_lane0, (int)matrix_id, address, data.u32);
          }
        }
      } else {
        // Transpose: load non-contiguous elements from different rows
        // For m8n8.b16: load 2 elements that are spec.rows apart
        // Element 0: (frag_col_base, frag_row) -> offset frag_col_base * 8 +
        // frag_row Element 1: (frag_col_base+1, frag_row) -> offset
        // (frag_col_base+1) * 8 + frag_row

        for (int elem = 0; elem < spec.cols_per_lane; elem++) {
          // Which row to access in the transposed view
          int target_row = matrix_col_id + elem;

          // Get base address from the lane providing this row's address
          auto address_lane_id = matrix_id * spec.rows + target_row;
          auto address_thread =
              core->get_thread_info()[tid_lane0 + address_lane_id];
          auto base_address = get_u32_value(address_thread, addr_operand);

          // Offset within the row (which is the original frag_row)
          uint32_t address = base_address + matrix_row_id * element_size;

          if constexpr (is_load) {
            thread->m_shared_mem->read(
                address, element_size,
                reinterpret_cast<char *>(&data.u16_2[elem]));
          } else {
            thread->m_shared_mem->write(
                address, element_size,
                reinterpret_cast<char *>(&data.u16_2[elem]), thread, pI);
          }
        }
      }

      GPPRINTF_INST_EXEC(WIP,
                         "%s: lane_id=%d, matrix_id=%d, "
                         "matrix_row_id=%d, matrix_col_id=%d, "
                         "data=%04x %04x (transpose=%d)\n",
                         inst_name, lane_id, matrix_id, matrix_row_id,
                         matrix_col_id, result_regs[matrix_id].u16_2[0],
                         result_regs[matrix_id].u16_2[1], is_transpose);
    }

    // For ldmatrix: write to the destination register after loading all
    // matrices
    if constexpr (is_load) {
      thread->set_vector_operand_values(pI->dst(), result_regs[0],
                                        result_regs[1], result_regs[2],
                                        result_regs[3]);
    }
  }
}

void handle_ldmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  handle_ld_st_matrix_inst_impl<MatrixDirection::LOAD>(pI, core, inst);
}

void handle_stmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  handle_ld_st_matrix_inst_impl<MatrixDirection::STORE>(pI, core, inst);
}
