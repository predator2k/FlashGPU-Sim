#ifndef FLASH_GPGPU_SIM_MBARRIER_H
#define FLASH_GPGPU_SIM_MBARRIER_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>

class gpgpu_sim;
namespace flash_gpgpu_sim {

enum class mbarrier_recheck_action_t {
  KEEP_SLEEPING,
  RETURN_TRUE,
  RETURN_FALSE,
};

uint64_t mbarrier_saturating_add(uint64_t cycle, uint64_t delta);
uint64_t mbarrier_hint_ns_to_cycles(uint32_t hint_ns, unsigned core_freq_hz);
mbarrier_recheck_action_t mbarrier_classify_recheck(bool phase_complete,
                                                    uint64_t cycle,
                                                    uint64_t deadline_cycle);
uint64_t mbarrier_wake_on_phase_notification(uint64_t scheduled_wake_cycle,
                                             uint64_t notification_cycle);
bool mbarrier_should_delay_phase_wakeup(bool suspended,
                                        bool phase_notification_pending,
                                        bool all_true, unsigned latency);
bool mbarrier_latches_initial_predicate(bool has_time_hint,
                                        unsigned predicate_latency);
uint64_t mbarrier_result_release_cycle(uint64_t issue_cycle,
                                       uint64_t resolution_cycle,
                                       unsigned predicate_latency,
                                       unsigned phase_wakeup_latency);

class mbarrier_manager_t {

  /**
   * Implement the mbarrier instruction.
   * NOTE: So far, this is more like a idealized implementation with some
   * limitations:
   * 1. It does not access the shared memory,
   * 2. only support CTA level synchronization.
   * 3. Does not support thread-level barrier (this is a limitation from
   * GPGPU-Sim, we can fix later). So far if one thread in a warp is blocked by
   * a barrier, the entire warp is blocked.
   */
  struct mbarrier_t {
    mbarrier_t(int id, int hw_cta_id, int sw_cta_id, uint64_t addr,
               int expected_count)
        : m_id(id), m_hw_cta_id(hw_cta_id), m_sw_cta_id(sw_cta_id),
          m_addr(addr), m_expected_count(expected_count),
          m_pending_arrival_count(expected_count), m_tx_count(0), m_phase(0) {}

    const int m_id;
    const int m_hw_cta_id;
    const int m_sw_cta_id;
    const uint64_t m_addr;
    const int m_expected_count;
    int m_pending_arrival_count;
    // This is for TMA interaction. It may change every phase.
    int m_tx_count;
    int m_phase;
    std::set<int> m_waiting_warps;
  };

public:
  mbarrier_manager_t() : m_next_id(0) {}

  mbarrier_t *get_mbarrier(int sw_cta_id, uint64_t addr) {
    auto it = addr_to_mbarrier_map.find(std::make_pair(sw_cta_id, addr));
    if (it != addr_to_mbarrier_map.end()) {
      return it->second.get();
    } else {
      return nullptr;
    }
  }

  void reset() {
    m_next_id = 0;
    addr_to_mbarrier_map.clear();
  }

  void dump() const;

  struct thread_index_t {
    int hw_cta_id;
    int hw_warp_id;
    int sw_cta_id;
    int sw_warp_id;
    int sm_id = -1;
  };

  void init(gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr,
            int expected_count);
  void inval(gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr);

  /** Query the current phase without registering or changing barrier state. */
  bool test_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
                 uint64_t addr, int parity) const;

  /** Register one warp for a phase-change notification. */
  void register_wait(const thread_index_t &thread_index, uint64_t addr);

  /** Remove a waiter after timeout or lifecycle cleanup. */
  void cancel_wait(int sw_cta_id, uint64_t addr, int hw_warp_id);

  /**
   * Arrive at the mbarrier at addr with arrival_count for warp warp_id.
   * @return the set of warp ids that are released due to this arrive.
   */
  std::set<int> arrive(gpgpu_sim *gpu, const thread_index_t &thread_index,
                       uint64_t addr, int arrival_count);

  // The default form reserves one pending arrival; .noinc only validates it.
  void prepare_async_arrival(gpgpu_sim *gpu, const thread_index_t &thread_index,
                             uint64_t addr, bool increment_pending);

  /**
   * Complete transaction at the mbarrier at addr with completed_tx_count for
   * warp warp_id.
   * @return the set of warp ids that are released due to this complete_tx.
   */
  std::set<int> complete_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
                            uint64_t addr, int completed_tx_count);

  /**
   * Increase the expected tx count for the mbarrier at addr.
   * @return void
   */
  void expect_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
                 uint64_t addr, int expected_tx_count);

  /**
   * Clean up all mbarriers for a given hw_cta_id when the CTA completes.
   * This prevents stale barriers when hw_cta_ids get recycled.
   */
  void cleanup_cta(unsigned hw_cta_id);

private:
  int m_next_id;
  struct pair_hash {
    size_t operator()(const std::pair<int, uint64_t> &p) const noexcept {
      return std::hash<int>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
    }
  };
  std::unordered_map<std::pair<int, uint64_t>, std::unique_ptr<mbarrier_t>,
                     pair_hash>
      addr_to_mbarrier_map;

  /**
   * Check if the mbarrier is satisfied, and release warps if so.
   * @return the set of warp ids that are released.
   */
  std::set<int> try_advance(gpgpu_sim *gpu, const thread_index_t &thread_index,
                            mbarrier_t *mbarrier);
};
} // namespace flash_gpgpu_sim

// Forward declarations for mbarrier_impl
class ptx_instruction;
class ptx_thread_info;
class operand_info;
void handle_mbarrier_inst(const ptx_instruction *pI, ptx_thread_info *thread);

// yasai: the shared-window address of the mbarrier operand op of pI, executed by thread: the value
// the forms with a state space (.shared, .shared::cta, .shared::cluster) carry, an offset in the
// CTA's shared memory with the owner's rank in bits 24 and up while a cluster executes
// (yasai_func::dsmem_encode). The forms without a state space take a generic address (PTX ISA,
// mbarrier: generic addressing; FlashInfer's XQA kernel_mha initialises its barriers with
// "mbarrier.init.b64 [%rd], n" on a cvta.shared address):
//   * a value below the window size (1 MiB, or 16 ranks of 16 MiB while a cluster executes) is
//     a window address already and is returned unchanged;
//   * a generic address in this CTA's generic shared window becomes its offset;
//   * while a cluster executes, a generic address in the window of another CTA of the cluster
//     becomes that CTA's .shared::cluster address (yasai_func::mbar_canonical maps it to the CTA);
//   * any other value stops the simulation with one yasai-strictaddr line in every
//     YASAI_STRICT_ADDR mode: it names no barrier, the GPU's behaviour is undefined, and a
//     simulation that went on would wait on or arrive at a barrier nobody initialised.
// The operand is read with 64 bits (a 32-bit address register keeps its 32 bits): reading 32 bits
// kept 0xC0000000 + smid * 1 MiB + offset of a generic address, a key other than the offset's, so
// a generic init and a .shared arrive at one barrier reached two barriers.
//   pI: the mbarrier, cp.async.mbarrier.arrive or tcgen05.commit instruction
//   thread: the executing thread
//   op: the barrier address operand of pI
//   return: the shared-window address (the key of the barrier with the CTA)
unsigned yasai_mbar_window_addr(const ptx_instruction *pI, ptx_thread_info *thread,
                                const operand_info &op);
#endif
