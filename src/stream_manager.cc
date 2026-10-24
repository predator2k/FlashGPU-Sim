// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#include "stream_manager.h"
#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpusim_entrypoint.h"

#include <unistd.h>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// yasai: the host-callback thread. cudaLaunchHostFunc operations are handed to it by the
// simulation thread when they reach the front of their stream; it calls the functions one at a
// time in that order (as the CUDA driver's callback thread does) and completes each operation
// on its stream when the function returns, so the stream's next operation can start. The
// simulation thread therefore never runs application code and never waits for it.
class yasai_host_callbacks {
 public:
  // Queues fn(user) for stream s.
  void post(void (*fn)(void *), void *user, CUstream_st *s) {
    {
      std::lock_guard<std::mutex> lk(m_mu);
      m_queue.push_back(item{fn, user, s});
    }
    m_cv.notify_one();
  }
  // The instance of this process; a forked child gets a new one with its own thread.
  static yasai_host_callbacks &get() {
    static yasai_host_callbacks *inst = NULL;
    static pid_t owner = 0;
    static std::mutex create_mu;
    std::lock_guard<std::mutex> lk(create_mu);
    if (inst == NULL || owner != getpid()) {
      inst = new yasai_host_callbacks();  // the parent's instance is left alone in a child
      owner = getpid();
    }
    return *inst;
  }

 private:
  struct item {
    void (*fn)(void *);  // host function
    void *user;          // its argument
    CUstream_st *stream;  // stream whose front operation it is
  };
  yasai_host_callbacks() {
    std::thread([this] { run(); }).detach();
  }
  // Calls the queued functions in order.
  void run() {
    for (;;) {
      item it;
      {
        std::unique_lock<std::mutex> lk(m_mu);
        m_cv.wait(lk, [this] { return !m_queue.empty(); });
        it = m_queue.front();
        m_queue.pop_front();
      }
      it.fn(it.user);
      it.stream->record_next_done();
    }
  }
  std::mutex m_mu;
  std::condition_variable m_cv;
  std::deque<item> m_queue;
};

// yasai: sleeps for a growing interval (20 us up to 1 ms) between polls of a host wait.
void yasai_backoff(unsigned *us) {
  usleep(*us);
  if (*us < 1000) *us *= 2;
}

}  // namespace

unsigned CUstream_st::sm_next_stream_uid = 0;

// SST memcpy callbacks
extern void SST_callback_memcpy_H2D_done();
extern void SST_callback_memcpy_D2H_done();
extern void SST_callback_memcpy_to_symbol_done();
extern void SST_callback_memcpy_from_symbol_done();
__attribute__((weak)) void SST_callback_memcpy_H2D_done() {}
__attribute__((weak)) void SST_callback_memcpy_D2H_done() {}
__attribute__((weak)) void SST_callback_memcpy_to_symbol_done() {}
__attribute__((weak)) void SST_callback_memcpy_from_symbol_done() {}

CUstream_st::CUstream_st() {
  m_pending = false;
  m_uid = sm_next_stream_uid++;
  pthread_mutex_init(&m_lock, NULL);
}

bool CUstream_st::empty() {
  pthread_mutex_lock(&m_lock);
  bool empty = m_operations.empty();
  pthread_mutex_unlock(&m_lock);
  return empty;
}

bool CUstream_st::busy() {
  pthread_mutex_lock(&m_lock);
  bool pending = m_pending;
  pthread_mutex_unlock(&m_lock);
  return pending;
}

void CUstream_st::synchronize() {
  // called by host thread
  bool done = false;
  do {
    pthread_mutex_lock(&m_lock);
    done = m_operations.empty();
    pthread_mutex_unlock(&m_lock);
  } while (!done);
}

bool CUstream_st::yasai_front_seq(uint64_t *seq) {
  pthread_mutex_lock(&m_lock);
  const bool any = !m_operations.empty();
  if (any) *seq = m_operations.front().yasai_seq();
  pthread_mutex_unlock(&m_lock);
  return any;
}

void CUstream_st::push(const stream_operation &op) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  m_operations.push_back(op);
  pthread_mutex_unlock(&m_lock);
}

void CUstream_st::record_next_done() {
  // called by gpu thread
  pthread_mutex_lock(&m_lock);
  assert(m_pending);
  m_operations.pop_front();
  m_pending = false;
  pthread_mutex_unlock(&m_lock);
}

stream_operation CUstream_st::next() {
  // called by gpu thread
  pthread_mutex_lock(&m_lock);
  m_pending = true;
  stream_operation result = m_operations.front();
  pthread_mutex_unlock(&m_lock);
  return result;
}

void CUstream_st::cancel_front() {
  pthread_mutex_lock(&m_lock);
  assert(m_pending);
  m_pending = false;
  pthread_mutex_unlock(&m_lock);
}

void CUstream_st::print(FILE *fp) {
  pthread_mutex_lock(&m_lock);
  fprintf(fp, "GPGPU-Sim API:    stream %u has %zu operations\n", m_uid,
          m_operations.size());
  std::list<stream_operation>::iterator i;
  unsigned n = 0;
  for (i = m_operations.begin(); i != m_operations.end(); i++) {
    stream_operation &op = *i;
    fprintf(fp, "GPGPU-Sim API:       %u : ", n++);
    op.print(fp);
    fprintf(fp, "\n");
  }
  pthread_mutex_unlock(&m_lock);
}

bool stream_operation::do_operation(gpgpu_sim *gpu) {
  if (is_noop()) return true;

  assert(!m_done && m_stream);
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim API: stream %u performing ", m_stream->get_uid());
  switch (m_type) {
    case stream_memcpy_host_to_device:
      if (g_debug_execution >= 3) printf("memcpy host-to-device\n");
      gpu->memcpy_to_gpu(m_device_address_dst, m_host_address_src, m_cnt);
      m_stream->record_next_done();
      if (gpu->is_SST_mode()) SST_callback_memcpy_H2D_done();
      break;
    case stream_memcpy_device_to_host:
      if (g_debug_execution >= 3) printf("memcpy device-to-host\n");
      gpu->memcpy_from_gpu(m_host_address_dst, m_device_address_src, m_cnt);
      m_stream->record_next_done();
      if (gpu->is_SST_mode()) SST_callback_memcpy_D2H_done();
      break;
    case stream_memset:
      gpu->gpu_memset(m_device_address_dst, m_memset_value, m_cnt);
      m_stream->record_next_done();
      break;
    case stream_host_fn:
      // yasai: completed by the host-callback thread when the function returns.
      yasai_host_callbacks::get().post(m_host_fn, m_host_fn_user, m_stream);
      break;
    case stream_memcpy_device_to_device:
      if (g_debug_execution >= 3) printf("memcpy device-to-device\n");
      gpu->memcpy_gpu_to_gpu(m_device_address_dst, m_device_address_src, m_cnt);
      m_stream->record_next_done();
      break;
    case stream_memcpy_to_symbol:
      if (g_debug_execution >= 3) printf("memcpy to symbol\n");
      gpu->gpgpu_ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol(
          m_symbol, m_host_address_src, m_cnt, m_offset, 1, gpu);
      m_stream->record_next_done();
      if (gpu->is_SST_mode()) SST_callback_memcpy_to_symbol_done();
      break;
    case stream_memcpy_from_symbol:
      if (g_debug_execution >= 3) printf("memcpy from symbol\n");
      gpu->gpgpu_ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol(
          m_symbol, m_host_address_dst, m_cnt, m_offset, 0, gpu);
      m_stream->record_next_done();
      if (gpu->is_SST_mode()) SST_callback_memcpy_from_symbol_done();
      break;
    case stream_kernel_launch:
      if (m_sim_mode) {  // Functional Sim
        if (g_debug_execution >= 3) {
          printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
                 m_kernel->get_uid(), m_kernel->name().c_str());
          m_kernel->print_parent_info();
        }
        gpu->set_cache_config(m_kernel->name());
        gpu->functional_launch(m_kernel);
      } else {  // Performance Sim
        if (gpu->can_start_kernel() && m_kernel->m_launch_latency == 0) {
          if (g_debug_execution >= 3) {
            printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
                   m_kernel->get_uid(), m_kernel->name().c_str());
            m_kernel->print_parent_info();
          }
          gpu->set_cache_config(m_kernel->name());
          gpu->launch(m_kernel);
        } else {
          if (m_kernel->m_launch_latency) m_kernel->m_launch_latency--;
          if (g_debug_execution >= 3)
            printf(
                "kernel %d: \'%s\', latency %u not ready to transfer to GPU "
                "hardware scheduler\n",
                m_kernel->get_uid(), m_kernel->name().c_str(),
                m_kernel->m_launch_latency);
          return false;
        }
      }
      break;
    case stream_event: {
      printf("event update\n");
      time_t wallclock = time((time_t *)NULL);
      m_event->update(gpu->gpu_tot_sim_cycle, wallclock);
      m_stream->record_next_done();
    } break;
    case stream_wait_event:
      // only allows next op to go if event is done
      // otherwise stays in the stream queue
      printf("stream wait event processing...\n");
      if (m_event->num_updates() >= m_cnt) {
        printf("stream wait event done\n");
        m_stream->record_next_done();
      } else {
        return false;
      }
      break;
    default:
      abort();
  }
  m_done = true;
  fflush(stdout);
  return true;
}

void stream_operation::print(FILE *fp) const {
  fprintf(fp, " stream operation ");
  switch (m_type) {
    case stream_event:
      fprintf(fp, "event");
      break;
    case stream_kernel_launch:
      fprintf(fp, "kernel");
      break;
    case stream_memset:
      fprintf(fp, "memset");
      break;
    case stream_memcpy_device_to_device:
      fprintf(fp, "memcpy device-to-device");
      break;
    case stream_memcpy_device_to_host:
      fprintf(fp, "memcpy device-to-host");
      break;
    case stream_memcpy_host_to_device:
      fprintf(fp, "memcpy host-to-device");
      break;
    case stream_memcpy_to_symbol:
      fprintf(fp, "memcpy to symbol");
      break;
    case stream_memcpy_from_symbol:
      fprintf(fp, "memcpy from symbol");
      break;
    case stream_no_op:
      fprintf(fp, "no-op");
      break;
    default:
      break;
  }
}

stream_manager::stream_manager(gpgpu_sim *gpu, bool cuda_launch_blocking) {
  m_gpu = gpu;
  m_service_stream_zero = false;
  m_cuda_launch_blocking = cuda_launch_blocking;
  const char *hs = getenv("YASAI_STREAM_ZERO_HOST_SYNC");
  m_yasai_host_sync = hs != NULL && atoi(hs) != 0;
  pthread_mutex_init(&m_lock, NULL);
  m_last_stream = m_streams.begin();
}

bool stream_manager::operation(bool *sim) {
  bool check = check_finished_kernel();
  pthread_mutex_lock(&m_lock);
  //    if(check)m_gpu->print_stats();
  stream_operation op = front();
  if (!op.do_operation(m_gpu))  // not ready to execute
  {
    // cancel operation
    if (op.is_kernel()) {
      unsigned grid_uid = op.get_kernel()->get_uid();
      m_grid_id_to_stream.erase(grid_uid);
    }
    op.get_stream()->cancel_front();
  }
  pthread_mutex_unlock(&m_lock);
  // pthread_mutex_lock(&m_lock);
  // simulate a clock cycle on the GPU
  return check;
}

bool stream_manager::check_finished_kernel() {
  unsigned grid_uid = m_gpu->finished_kernel();
  bool check = register_finished_kernel(grid_uid);
  return check;
}

bool stream_manager::register_finished_kernel(unsigned grid_uid) {
  // called by gpu simulation thread
  if (grid_uid > 0) {
    CUstream_st *stream = m_grid_id_to_stream[grid_uid];
    kernel_info_t *kernel = stream->front().get_kernel();
    assert(grid_uid == kernel->get_uid());

    // Jin: should check children kernels for CDP
    if (kernel->is_finished()) {
      //            std::ofstream kernel_stat("kernel_stat.txt",
      //            std::ofstream::out | std::ofstream::app); kernel_stat<< "
      //            kernel " << grid_uid << ": " << kernel->name();
      //            if(kernel->get_parent())
      //                kernel_stat << ", parent " <<
      //                kernel->get_parent()->get_uid() <<
      //                ", launch " << kernel->launch_cycle;
      //            kernel_stat<< ", start " << kernel->start_cycle <<
      //                ", end " << kernel->end_cycle << ", retire " <<
      //                gpu_sim_cycle + gpu_tot_sim_cycle << "\n";
      //            printf("kernel %d finishes, retires from stream %d\n",
      //            grid_uid, stream->get_uid()); kernel_stat.flush();
      //            kernel_stat.close();
      stream->record_next_done();
      m_grid_id_to_stream.erase(grid_uid);
      kernel->notify_parent_finished();
      delete kernel;
      return true;
    }
  }

  return false;
}

void stream_manager::stop_all_running_kernels() {
  pthread_mutex_lock(&m_lock);

  std::vector<unsigned long long> finished_streams;
  std::vector<kernel_info_t *> running_kernels = m_gpu->get_running_kernels();
  for (kernel_info_t *k : running_kernels) {
    if (k != NULL) {
      finished_streams.push_back(k->get_streamID());
    }
  }

  // Signal m_gpu to stop all running kernels
  m_gpu->stop_all_running_kernels();

  // Clean up all streams waiting on running kernels
  int count = 0;
  while (check_finished_kernel()) {
    count++;
  }

  // If any kernels completed, print out the current stats
  for (unsigned long long streamID : finished_streams) {
    m_gpu->print_stats(streamID);
  }

  pthread_mutex_unlock(&m_lock);
}

bool stream_manager::yasai_eligible(CUstream_st *s, uint64_t seq) {
  if (s->yasai_nonblocking()) return true;
  uint64_t f = 0;
  if (s == &m_stream_zero) {
    for (CUstream_st *b : m_streams)
      if (!b->yasai_nonblocking() && b->yasai_front_seq(&f) && f < seq) return false;
    return true;
  }
  return !(m_stream_zero.yasai_front_seq(&f) && f < seq);
}

stream_operation stream_manager::front() {
  // called by gpu simulation thread
  stream_operation result;
  // yasai: legacy default stream semantics on the device side. The streams (the legacy one
  // first) are scanned round-robin from the one after the last served, and the first idle
  // stream whose front operation is eligible (yasai_eligible) provides the next operation.
  if (!m_cuda_launch_blocking && !m_yasai_host_sync) {
    std::vector<CUstream_st *> c;
    c.reserve(m_streams.size() + 1);
    c.push_back(&m_stream_zero);
    for (CUstream_st *s : m_streams) c.push_back(s);
    const size_t n = c.size();
    for (size_t i = 1; i <= n; i++) {
      const size_t idx = (m_yasai_rr + i) % n;
      CUstream_st *s = c[idx];
      uint64_t f = 0;
      if (s->busy() || !s->yasai_front_seq(&f) || !yasai_eligible(s, f)) continue;
      m_yasai_rr = idx;
      result = s->next();
      if (result.is_kernel()) m_grid_id_to_stream[result.get_kernel()->get_uid()] = s;
      break;
    }
    return result;
  }
  //    if( concurrent_streams_empty() )
  m_service_stream_zero = true;
  if (m_service_stream_zero) {
    if (!m_stream_zero.empty() && !m_stream_zero.busy()) {
      result = m_stream_zero.next();
      if (result.is_kernel()) {
        unsigned grid_id = result.get_kernel()->get_uid();
        m_grid_id_to_stream[grid_id] = &m_stream_zero;
      }
    } else {
      m_service_stream_zero = false;
    }
  }
  if (!m_service_stream_zero) {
    std::list<struct CUstream_st *>::iterator s = m_last_stream;
    if (m_last_stream == m_streams.end()) {
      s = m_streams.begin();
    } else {
      s++;
    }
    for (size_t ii = 0; ii < m_streams.size(); ii++, s++) {
      if (s == m_streams.end()) {
        s = m_streams.begin();
      }
      m_last_stream = s;
      CUstream_st *stream = *s;
      if (!stream->busy() && !stream->empty()) {
        result = stream->next();
        if (result.is_kernel()) {
          unsigned grid_id = result.get_kernel()->get_uid();
          m_grid_id_to_stream[grid_id] = stream;
        }
        break;
      }
    }
  }
  return result;
}

void stream_manager::add_stream(struct CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  m_streams.push_back(stream);
  pthread_mutex_unlock(&m_lock);
}

void stream_manager::destroy_stream(CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  while (!stream->empty());
  std::list<CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    if (*s == stream) {
      m_streams.erase(s);
      break;
    }
  }
  delete stream;
  m_last_stream = m_streams.begin();
  m_yasai_rr = 0;
  pthread_mutex_unlock(&m_lock);
}

bool stream_manager::concurrent_streams_empty() {
  bool result = true;
  if (m_streams.empty()) return true;
  // called by gpu simulation thread
  std::list<struct CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    struct CUstream_st *stream = *s;
    if (!stream->empty()) {
      // stream->print(stdout);
      result = false;
      break;
    }
  }
  return result;
}

bool stream_manager::empty_protected() {
  bool result = true;
  pthread_mutex_lock(&m_lock);
  if (!concurrent_streams_empty()) result = false;
  if (!m_stream_zero.empty()) result = false;
  pthread_mutex_unlock(&m_lock);
  return result;
}

bool stream_manager::empty() {
  bool result = true;
  if (!concurrent_streams_empty()) result = false;
  if (!m_stream_zero.empty()) result = false;
  return result;
}

void stream_manager::print(FILE *fp) {
  pthread_mutex_lock(&m_lock);
  print_impl(fp);
  pthread_mutex_unlock(&m_lock);
}
void stream_manager::print_impl(FILE *fp) {
  fprintf(fp, "GPGPU-Sim API: Stream Manager State\n");
  std::list<struct CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    struct CUstream_st *stream = *s;
    if (!stream->empty()) stream->print(fp);
  }
  if (!m_stream_zero.empty()) m_stream_zero.print(fp);
}

CUstream_st *stream_manager::yasai_resolve(CUstream_st *s) {
  const uintptr_t h = (uintptr_t)s;
  if (h == 0 || h == 1) return NULL;  // NULL, cudaStreamLegacy
  if (h != 2) return s;               // cudaStreamPerThread is 0x2
  pthread_mutex_lock(&m_lock);
  CUstream_st *&mine = m_yasai_per_thread[pthread_self()];
  if (mine == NULL) {
    mine = new CUstream_st();  // a blocking stream of the calling thread, never destroyed
    m_streams.push_back(mine);
  }
  CUstream_st *r = mine;
  pthread_mutex_unlock(&m_lock);
  return r;
}

void stream_manager::push_sync(stream_operation op) {
  CUstream_st *s = yasai_resolve(op.get_stream());
  const uint64_t seq = push(op);
  CUstream_st *target = (s && !m_cuda_launch_blocking) ? s : &m_stream_zero;
  unsigned us = 20;
  uint64_t f = 0;
  while (target->yasai_front_seq(&f) && f <= seq) yasai_backoff(&us);
}

void stream_manager::yasai_wait_stream(CUstream_st *handle, uint64_t seq) {
  CUstream_st *s = yasai_resolve(handle);
  pthread_mutex_lock(&m_lock);
  if (seq == UINT64_MAX) seq = m_yasai_next_seq - 1;
  pthread_mutex_unlock(&m_lock);
  unsigned us = 20;
  for (;;) {
    bool busy = false;
    uint64_t f = 0;
    pthread_mutex_lock(&m_lock);
    if (s != NULL) {
      busy = s->yasai_front_seq(&f) && f <= seq;
    } else {
      busy = m_stream_zero.yasai_front_seq(&f) && f <= seq;
      for (CUstream_st *b : m_streams)
        if (!busy && !b->yasai_nonblocking() && b->yasai_front_seq(&f) && f <= seq) busy = true;
    }
    pthread_mutex_unlock(&m_lock);
    if (!busy) return;
    yasai_backoff(&us);
  }
}

uint64_t stream_manager::push(stream_operation op) {
  struct CUstream_st *stream = yasai_resolve(op.get_stream());
  op.set_stream(stream);
  if (!m_cuda_launch_blocking && !m_yasai_host_sync) {
    // yasai: no host wait; see front() and yasai_eligible.
    pthread_mutex_lock(&m_lock);
    const uint64_t seq = m_yasai_next_seq++;
    op.yasai_set_seq(seq);
    if (!m_gpu->cycle_insn_cta_max_hit()) {
      if (stream) {
        stream->push(op);
      } else {
        op.set_stream(&m_stream_zero);
        m_stream_zero.push(op);
      }
    } else {
      printf("GPGPU-Sim API: Maximum cycle, instruction, or CTA count hit. Skipping:");
      op.print(stdout);
      printf("\n");
    }
    pthread_mutex_unlock(&m_lock);
    return seq;
  }

  // block if stream 0 (or concurrency disabled) and pending concurrent
  // operations exist
  bool block = !stream || m_cuda_launch_blocking;
  while (block) {
    pthread_mutex_lock(&m_lock);
    block = !concurrent_streams_empty();
    pthread_mutex_unlock(&m_lock);
  };

  pthread_mutex_lock(&m_lock);
  const uint64_t seq = m_yasai_next_seq++;
  op.yasai_set_seq(seq);
  if (!m_gpu->cycle_insn_cta_max_hit()) {
    // Accept the stream operation if the maximum cycle/instruction/cta counts
    // are not triggered
    if (stream && !m_cuda_launch_blocking) {
      stream->push(op);
    } else {
      op.set_stream(&m_stream_zero);
      m_stream_zero.push(op);
    }
  } else {
    // Otherwise, ignore operation and continue
    printf(
        "GPGPU-Sim API: Maximum cycle, instruction, or CTA count hit. "
        "Skipping:");
    op.print(stdout);
    printf("\n");
  }
  if (g_debug_execution >= 3) print_impl(stdout);
  pthread_mutex_unlock(&m_lock);
  if (!m_gpu->is_SST_mode() && (m_cuda_launch_blocking || stream == NULL)) {
    unsigned int wait_amount = 100;
    unsigned int wait_cap = 100000;  // 100ms
    while (!empty()) {
      // sleep to prevent CPU hog by empty spin
      // sleep time increased exponentially ensure fast response when needed
      usleep(wait_amount);
      wait_amount *= 2;
      if (wait_amount > wait_cap) wait_amount = wait_cap;
    }
  }
  return seq;
}

void stream_manager::pushCudaStreamWaitEventToAllStreams(CUevent_st *e,
                                                         unsigned int flags) {
  std::list<CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    stream_operation op(*s, e, flags);
    push(op);
  }
}
