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
#include "cuda-sim/yasai_lazy.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpusim_entrypoint.h"
#include "yasai_realtime.h"
#include "cuda-sim/fio_window.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_ir.h"
#include "kernel_info.h"

#include <unistd.h>
#include <atomic>
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

// yasai: YASAI_STREAM_TRACE=1 prints every push and every issue of a stream operation.
bool yasai_stream_trace() {
  static const bool on = getenv("YASAI_STREAM_TRACE") != NULL && atoi(getenv("YASAI_STREAM_TRACE")) != 0;
  return on;
}

// yasai: sleeps for a growing interval (20 us up to 1 ms) between polls of a host wait.
void yasai_backoff(unsigned *us) {
  yasai_real_usleep(*us);
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
    // yasai: sleep between polls, so the waiting host thread leaves the core to the simulation
    // thread that executes the stream (one core per rank under taskset)
    if (!done) yasai_real_usleep(100);
  } while (!done);
}

bool CUstream_st::yasai_front_seq(uint64_t *seq) {
  pthread_mutex_lock(&m_lock);
  const bool any = !m_operations.empty();
  if (any) *seq = m_operations.front().yasai_seq();
  pthread_mutex_unlock(&m_lock);
  return any;
}

bool CUstream_st::yasai_front_ready() {
  pthread_mutex_lock(&m_lock);
  const bool r = !m_pending && !m_operations.empty() && m_operations.front().yasai_ready();
  pthread_mutex_unlock(&m_lock);
  return r;
}

bool stream_operation::yasai_ready() const {
  if (m_type == stream_wait_event) return m_event->num_updates() >= m_cnt;
  return true;
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
  g_yasai_device_progress.fetch_add(1, std::memory_order_relaxed);
}

void CUstream_st::yasai_count_pending(unsigned *ops, std::set<uint64_t> *groups) {
  pthread_mutex_lock(&m_lock);
  for (const stream_operation &op : m_operations) {
    if (!op.yasai_limited()) continue;
    if (op.yasai_group() != 0)
      groups->insert(op.yasai_group());
    else
      ++*ops;
  }
  pthread_mutex_unlock(&m_lock);
}

bool CUstream_st::yasai_print_front(FILE *fp) {
  pthread_mutex_lock(&m_lock);
  const bool any = !m_operations.empty();
  if (any) {
    fprintf(fp, "stream %u%s: ", m_uid, m_pending ? " (running)" : "");
    m_operations.front().print(fp);
  }
  pthread_mutex_unlock(&m_lock);
  return any;
}

std::atomic<unsigned long long> g_yasai_device_progress{0};

namespace {
/// yasai: the cudaGraphLaunch whose nodes the calling thread queues (0: none), and whether its
/// first operation passed the host-ahead limit already (yasai_graph_launch_begin).
thread_local uint64_t t_yasai_graph_group = 0;
thread_local bool t_yasai_graph_checked = false;
std::atomic<uint64_t> g_yasai_next_graph_group{1};
/// yasai: wall seconds a host thread waits at the host-ahead limit, with no device progress,
/// before the warning line.
constexpr double kWarnS = 600;
/// yasai: the manager whose host-ahead limit summary is printed at exit (the first with the limit).
stream_manager *g_yasai_limit_summary_of = NULL;
void yasai_limit_summary_at_exit() {
  if (g_yasai_limit_summary_of != NULL) g_yasai_limit_summary_of->yasai_limit_summary();
}
}  // namespace

void yasai_graph_launch_begin() {
  t_yasai_graph_group = g_yasai_next_graph_group.fetch_add(1, std::memory_order_relaxed);
  t_yasai_graph_checked = false;
}

void yasai_graph_launch_end() {
  t_yasai_graph_group = 0;
  t_yasai_graph_checked = false;
}

unsigned stream_manager::yasai_pending_ops() {
  unsigned ops = 0;
  std::set<uint64_t> groups;
  pthread_mutex_lock(&m_lock);
  m_stream_zero.yasai_count_pending(&ops, &groups);
  for (CUstream_st *s : m_streams) s->yasai_count_pending(&ops, &groups);
  pthread_mutex_unlock(&m_lock);
  return ops + (unsigned)groups.size();
}

void stream_manager::yasai_limit_summary() {
  printf("yasai-stream: host-ahead limit %u: most queued %u, waits %llu\n", m_yasai_max_pending,
         m_yasai_max_pending_seen.load(), m_yasai_limit_waits.load());
  fflush(stdout);
}

void stream_manager::yasai_limit_pending(const stream_operation &op) {
  if (m_yasai_max_pending == 0 || !op.yasai_limited()) return;
  if (t_yasai_graph_group != 0) {
    if (t_yasai_graph_checked) return;  // later nodes of a graph launch that passed
    t_yasai_graph_checked = true;
  }
  unsigned n = yasai_pending_ops();
  if (n < m_yasai_max_pending) return;
  m_yasai_limit_waits.fetch_add(1, std::memory_order_relaxed);
  struct timespec t0;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  double quiet_since = t0.tv_sec + t0.tv_nsec * 1e-9;
  unsigned long long progress = g_yasai_device_progress.load(std::memory_order_relaxed);
  bool warned = false;
  unsigned us = 20;
  while ((n = yasai_pending_ops()) >= m_yasai_max_pending) {
    yasai_backoff(&us);
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    const double now = t.tv_sec + t.tv_nsec * 1e-9;
    const unsigned long long p = g_yasai_device_progress.load(std::memory_order_relaxed);
    if (p != progress) {
      progress = p;
      quiet_since = now;
      warned = false;
    } else if (!warned && now - quiet_since > kWarnS) {
      // One line per quiet stretch: the fronts of the streams, then the operation that waits.
      printf("yasai-stream: warning: a host thread has waited %.0f s at YASAI_MAX_PENDING_LAUNCHES=%u (%u queued) "
             "while the device completed nothing; fronts:",
             now - quiet_since, m_yasai_max_pending, n);
      pthread_mutex_lock(&m_lock);
      printf(" [");
      m_stream_zero.yasai_print_front(stdout);
      printf("]");
      for (CUstream_st *s : m_streams) {
        printf(" [");
        s->yasai_print_front(stdout);
        printf("]");
      }
      pthread_mutex_unlock(&m_lock);
      printf("; waiting to queue: ");
      op.print(stdout);
      printf("\n");
      fflush(stdout);
      warned = true;
    }
  }
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
  fprintf(fp, "GPGPU-Sim API:    stream %u has %zu operations%s%s\n", m_uid,
          m_operations.size(), m_pending ? " (front running)" : "",
          m_yasai_nonblocking ? " (non-blocking)" : "");
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
      // yasai: a host function may read any device memory: every pending deferred kernel runs
      // first (yasai_lazy.h).
      yasai_lazy::observe_all("host function");
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
        fio_window::on_kernel_start();  // yasai: the GPU is busy (FIO window idle stepping)
      } else {  // Performance Sim
        if (gpu->can_start_kernel() && m_kernel->m_launch_latency == 0) {
          if (g_debug_execution >= 3) {
            printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
                   m_kernel->get_uid(), m_kernel->name().c_str());
            m_kernel->print_parent_info();
          }
          gpu->set_cache_config(m_kernel->name());
          gpu->launch(m_kernel);
          fio_window::on_kernel_start();  // yasai: the GPU is busy (FIO window idle stepping)
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
      // yasai: the monotonic clock goes through libc (clock_gettime), like the calendar clock
      // above, so the application's view of time (and a time-dilation preload) applies to it.
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      m_event->update((double)(gpu->gpu_tot_sim_cycle + gpu->gpu_sim_cycle), wallclock,
                      (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec,
                      yasai_kernels_finished(), m_stream->get_uid(),
                      m_stream->yasai_kernels_done());
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
    case stream_host_fn:
      fprintf(fp, "host function");
      break;
    case stream_wait_event:
      fprintf(fp, "wait event (needs %zu updates, has %d)", m_cnt,
              m_event ? (int)m_event->num_updates() : -1);
      break;
    default:
      break;
  }
  fprintf(fp, " seq %llu", (unsigned long long)m_yasai_seq);
}

stream_manager::stream_manager(gpgpu_sim *gpu, bool cuda_launch_blocking) {
  m_gpu = gpu;
  m_service_stream_zero = false;
  m_cuda_launch_blocking = cuda_launch_blocking;
  const char *hs = getenv("YASAI_STREAM_ZERO_HOST_SYNC");
  m_yasai_host_sync = hs != NULL && atoi(hs) != 0;
  // yasai: the host-ahead limit (yasai_limit_pending); unset, empty or 0 is no limit, the former
  // behaviour.
  const char *mp = getenv("YASAI_MAX_PENDING_LAUNCHES");
  if (mp != NULL && *mp != '\0') {
    char *end = NULL;
    const long k = strtol(mp, &end, 10);
    if (end == mp || *end != '\0' || k < 0) {
      fprintf(stderr, "yasai-stream: YASAI_MAX_PENDING_LAUNCHES=%s is not a count (0 or more)\n", mp);
      abort();
    }
    m_yasai_max_pending = (unsigned)k;
  }
  if (m_yasai_max_pending > 0) {
    printf("yasai-stream: a host thread waits while %u or more stream operations are queued and not completed "
           "(YASAI_MAX_PENDING_LAUNCHES)\n",
           m_yasai_max_pending);
    if (g_yasai_limit_summary_of == NULL) {
      g_yasai_limit_summary_of = this;
      atexit(yasai_limit_summary_at_exit);
    }
  }
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

// yasai: kernels finished in this process (cudaEventElapsedTime: an interval between two
// events that contains a kernel is never reported as 0).
static std::atomic<unsigned long long> g_yasai_kernels_finished{0};
unsigned long long yasai_kernels_finished() {
  return g_yasai_kernels_finished.load(std::memory_order_acquire);
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
      // yasai: counted before the stream advances, so an event recorded after the kernel sees it.
      g_yasai_kernels_finished.fetch_add(1, std::memory_order_acq_rel);
      stream->yasai_note_kernel_done();
      // yasai: a kernel end (FIO window: idle stepping once no kernel runs, or delivery of every
      // pending write in the kernel-end-flush mode), before the host can see the completion.
      fio_window::on_kernel_end();
      stream->record_next_done();
      m_grid_id_to_stream.erase(grid_uid);
      // yasai: a deferred launch stays alive; lazy execution runs and deletes it.
      if (kernel->yasai_lazy_keep()) return true;
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
  // yasai: legacy default stream semantics on the device side. Of the idle streams whose front
  // operation can run (no unmet event wait) and is eligible (yasai_eligible), the one whose
  // front operation was submitted first provides the next operation, so operations of
  // independent streams run in submission order, as an otherwise idle GPU runs them (NCCL's
  // connection setup relies on that order between its streams). A wait whose event is not yet
  // recorded no longer takes a turn.
  if (!m_cuda_launch_blocking && !m_yasai_host_sync) {
    CUstream_st *best = NULL;
    uint64_t best_seq = UINT64_MAX;
    auto consider = [&](CUstream_st *s) {
      uint64_t f = 0;
      if (!s->yasai_front_seq(&f) || f >= best_seq) return;
      if (!s->yasai_front_ready() || !yasai_eligible(s, f)) return;
      best = s;
      best_seq = f;
    };
    consider(&m_stream_zero);
    for (CUstream_st *s : m_streams) consider(s);
    if (best != NULL) {
      CUstream_st *s = best;
      result = s->next();
      if (yasai_stream_trace()) {
        printf("yasai-stream: issue stream %u%s:", s->get_uid(), s->yasai_nonblocking() ? " (nb)" : "");
        result.print(stdout);
        printf("\n");
      }
      if (result.is_kernel()) m_grid_id_to_stream[result.get_kernel()->get_uid()] = s;
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

bool stream_operation::yasai_uses(uint64_t lo, uint64_t hi, bool *kernel) {
  *kernel = false;
  auto overlaps = [&](uint64_t a, size_t n) { return n && a < hi && a + n > lo; };
  switch (m_type) {
    case stream_memcpy_host_to_device:
    case stream_memset:
      return overlaps(m_device_address_dst, m_cnt);
    case stream_memcpy_device_to_host:
      return overlaps(m_device_address_src, m_cnt);
    case stream_memcpy_device_to_device:
      return overlaps(m_device_address_src, m_cnt) || overlaps(m_device_address_dst, m_cnt);
    case stream_kernel_launch: {
      if (!m_kernel || !m_kernel->entry()) return false;
      *kernel = true;
      const function_info *f = m_kernel->entry();
      for (unsigned i = 0; i < f->num_args(); i++) {
        const symbol *s = f->get_arg(i);
        if (!s->has_valid_address()) continue;
        const uint64_t a0 = s->get_address(), a1 = a0 + s->get_size_in_bytes();
        for (uint64_t o = (a0 + 7) & ~7ull; o + 8 <= a1; o += 8) {
          uint64_t v = 0;
          m_kernel->get_param_memory()->read(o, 8, &v);
          if (v >= lo && v < hi) return true;
        }
      }
      return false;
    }
    default:
      return false;
  }
}

void CUstream_st::yasai_count_users(uint64_t lo, uint64_t hi, unsigned *copies,
                                    unsigned *kernels) {
  pthread_mutex_lock(&m_lock);
  for (stream_operation &op : m_operations) {
    bool k = false;
    if (op.yasai_uses(lo, hi, &k)) ++*(k ? kernels : copies);
  }
  pthread_mutex_unlock(&m_lock);
}

void stream_manager::yasai_count_users(uint64_t lo, uint64_t hi, unsigned *copies,
                                       unsigned *kernels) {
  pthread_mutex_lock(&m_lock);
  m_stream_zero.yasai_count_users(lo, hi, copies, kernels);
  for (CUstream_st *s : m_streams) s->yasai_count_users(lo, hi, copies, kernels);
  pthread_mutex_unlock(&m_lock);
}

void stream_manager::yasai_wait_device() {
  pthread_mutex_lock(&m_lock);
  const uint64_t seq = m_yasai_next_seq - 1;
  pthread_mutex_unlock(&m_lock);
  unsigned us = 20;
  for (;;) {
    uint64_t f = 0;
    pthread_mutex_lock(&m_lock);
    bool busy = m_stream_zero.yasai_front_seq(&f) && f <= seq;
    for (CUstream_st *b : m_streams)
      if (!busy && b->yasai_front_seq(&f) && f <= seq) busy = true;
    pthread_mutex_unlock(&m_lock);
    if (!busy) return;
    yasai_backoff(&us);
  }
}

uint64_t stream_manager::push(stream_operation op) {
  struct CUstream_st *stream = yasai_resolve(op.get_stream());
  op.set_stream(stream);
  // yasai: the operations of one cudaGraphLaunch count as one for the host-ahead limit, which the
  // calling thread passes before the operation is queued (yasai_limit_pending).
  if (t_yasai_graph_group != 0) op.yasai_set_group(t_yasai_graph_group);
  // The check and the queueing happen under one mutex, so threads that pass the check together
  // never queue more than the limit; the lock is released when the operation is queued.
  std::unique_lock<std::mutex> yasai_limit_lock(m_yasai_limit_mu, std::defer_lock);
  const bool yasai_limited = m_yasai_max_pending > 0 && op.yasai_limited();
  if (yasai_limited) {
    yasai_limit_lock.lock();
    yasai_limit_pending(op);
  }
  // Records the most counted operations queued, then lets the next thread check.
  auto yasai_queued = [&]() {
    if (!yasai_limited) return;
    const unsigned n = yasai_pending_ops();
    unsigned seen = m_yasai_max_pending_seen.load(std::memory_order_relaxed);
    while (n > seen && !m_yasai_max_pending_seen.compare_exchange_weak(seen, n)) {
    }
    yasai_limit_lock.unlock();
  };
  if (!m_cuda_launch_blocking && !m_yasai_host_sync) {
    // yasai: no host wait; see front() and yasai_eligible.
    pthread_mutex_lock(&m_lock);
    const uint64_t seq = m_yasai_next_seq++;
    op.yasai_set_seq(seq);
    if (yasai_stream_trace()) {
      printf("yasai-stream: push stream %u:", stream ? stream->get_uid() : m_stream_zero.get_uid());
      op.print(stdout);
      printf("\n");
    }
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
    yasai_queued();
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
  yasai_queued();
  if (!m_gpu->is_SST_mode() && (m_cuda_launch_blocking || stream == NULL)) {
    unsigned int wait_amount = 100;
    unsigned int wait_cap = 100000;  // 100ms
    while (!empty()) {
      // sleep to prevent CPU hog by empty spin
      // sleep time increased exponentially ensure fast response when needed
      yasai_real_usleep(wait_amount);
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
