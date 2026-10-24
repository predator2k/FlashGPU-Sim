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

#ifndef STREAM_MANAGER_H_INCLUDED
#define STREAM_MANAGER_H_INCLUDED

#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <list>
#include <map>
#include <memory>
#include "abstract_hardware_model.h"

// class stream_barrier {
// public:
//    stream_barrier() { m_pending_streams=0; }
//    void inc() { m_pending_streams++; }
//    void dec() { assert(m_pending_streams); m_pending_streams--; }
//    unsigned value() const { return m_pending_streams; }
// private:
//    unsigned m_pending_streams;
//};

struct CUevent_st {
 public:
  CUevent_st(bool blocking) {
    m_uid = ++m_next_event_uid;
    m_blocking = blocking;
    m_updates = 0;
    m_wallclock = 0;
    m_gpu_tot_sim_cycle = 0;
    m_issued = 0;
    m_done = false;
  }
  void update(double cycle, time_t clk) {
    if (m_shared) __atomic_fetch_add(&m_shared[1], 1u, __ATOMIC_ACQ_REL);
    else m_updates++;
    m_wallclock = clk;
    m_gpu_tot_sim_cycle = cycle;
    m_done = true;
  }
  // void set_done() { assert(!m_done); m_done=true; }
  int get_uid() const { return m_uid; }
  unsigned num_updates() const {
    return m_shared ? __atomic_load_n(&m_shared[1], __ATOMIC_ACQUIRE) : m_updates;
  }
  bool done() const { return num_updates() == num_issued(); }
  time_t clock() const { return m_wallclock; }
  void issue() {
    if (m_shared) __atomic_fetch_add(&m_shared[0], 1u, __ATOMIC_ACQ_REL);
    else m_issued++;
  }
  unsigned int num_issued() const {
    return m_shared ? __atomic_load_n(&m_shared[0], __ATOMIC_ACQUIRE) : m_issued;
  }
  /// yasai: makes the event interprocess: its issue and update counts live in shared memory
  /// (two 32-bit counters: [0] issued, [1] updates) that peer processes map through
  /// cudaIpcOpenEventHandle. The segment name goes into the IPC event handle.
  /// @param counters The two counters in a shared mapping.
  /// @param name Name of the shared memory segment.
  void set_shared(unsigned int *counters, const char *name) {
    m_shared = counters;
    snprintf(m_shared_name, sizeof(m_shared_name), "%s", name);
  }
  /// yasai: name of the shared segment of an interprocess event ("" otherwise).
  const char *shared_name() const { return m_shared_name; }

 private:
  unsigned int *m_shared = nullptr;  ///< yasai: shared counters of an interprocess event
  char m_shared_name[64] = {0};      ///< yasai: their segment name
  int m_uid;
  bool m_blocking;
  bool m_done;
  unsigned int m_updates;
  unsigned int m_issued;
  time_t m_wallclock;
  double m_gpu_tot_sim_cycle;

  static int m_next_event_uid;
};

enum stream_operation_type {
  stream_no_op,
  stream_memcpy_host_to_device,
  stream_memcpy_device_to_host,
  stream_memcpy_device_to_device,
  stream_memcpy_to_symbol,
  stream_memcpy_from_symbol,
  stream_kernel_launch,
  stream_event,
  stream_wait_event,
  stream_memset,  // yasai: cudaMemsetAsync, performed in stream order
  stream_host_fn  // yasai: cudaLaunchHostFunc, run by the host-callback thread in stream order
};

class stream_operation {
 public:
  /// yasai: a memset of cnt bytes of device memory at dst to value, in stream order.
  /// @param dst Device address.
  /// @param value Byte value.
  /// @param cnt Bytes.
  /// @param stream Stream.
  /// @return The operation.
  static stream_operation memset_op(size_t dst, int value, size_t cnt, struct CUstream_st *stream) {
    stream_operation op(size_t(0), dst, cnt, stream);
    op.m_type = stream_memset;
    op.m_memset_value = value;
    return op;
  }
  /// yasai: a host function call fn(user) in stream order (cudaLaunchHostFunc). The simulation
  /// thread hands it to the host-callback thread, and the stream's next operation starts when
  /// the function has returned.
  /// @param fn Host function.
  /// @param user Its argument.
  /// @param stream Stream.
  /// @return The operation.
  static stream_operation host_fn_op(void (*fn)(void *), void *user, struct CUstream_st *stream) {
    stream_operation op;
    op.m_type = stream_host_fn;
    op.m_host_fn = fn;
    op.m_host_fn_user = user;
    op.m_stream = stream;
    op.m_done = false;
    return op;
  }
  stream_operation() {
    m_kernel = NULL;
    m_type = stream_no_op;
    m_stream = NULL;
    m_done = true;
  }
  stream_operation(const void *src, const char *symbol, size_t count,
                   size_t offset, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_stream = stream;
    m_type = stream_memcpy_to_symbol;
    m_host_address_src = src;
    m_symbol = symbol;
    m_cnt = count;
    m_offset = offset;
    m_done = false;
  }
  stream_operation(const char *symbol, void *dst, size_t count, size_t offset,
                   struct CUstream_st *stream) {
    m_kernel = NULL;
    m_stream = stream;
    m_type = stream_memcpy_from_symbol;
    m_host_address_dst = dst;
    m_symbol = symbol;
    m_cnt = count;
    m_offset = offset;
    m_done = false;
  }
  stream_operation(kernel_info_t *kernel, bool sim_mode,
                   struct CUstream_st *stream) {
    m_type = stream_kernel_launch;
    m_kernel = kernel;
    m_sim_mode = sim_mode;
    m_stream = stream;
    m_done = false;
  }
  stream_operation(struct CUevent_st *e, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_event;
    m_event = e;
    m_stream = stream;
    m_done = false;
  }
  stream_operation(struct CUstream_st *stream, class CUevent_st *e,
                   unsigned int flags) {
    m_kernel = NULL;
    m_type = stream_wait_event;
    m_event = e;
    m_cnt = m_event->num_issued();
    m_stream = stream;
    m_done = false;
  }
  stream_operation(const void *host_address_src, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_memcpy_host_to_device;
    m_host_address_src = host_address_src;
    m_device_address_dst = device_address_dst;
    m_host_address_dst = NULL;
    m_device_address_src = 0;
    m_cnt = cnt;
    m_stream = stream;
    m_sim_mode = false;
    m_done = false;
  }
  /// yasai: a host-to-device copy from a snapshot buffer that the operation owns. Copies of
  /// the operation (the stream queues hold them by value) share the buffer, which is released
  /// when the last copy is destroyed, after the operation has run.
  /// @param snapshot Copy of the pageable host source, taken when the copy was issued.
  /// @param device_address_dst Device destination address.
  /// @param cnt Bytes.
  /// @param stream Stream of the copy.
  stream_operation(std::shared_ptr<const unsigned char[]> snapshot, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream)
      : stream_operation(static_cast<const void *>(snapshot.get()), device_address_dst, cnt,
                         stream) {
    m_snapshot = std::move(snapshot);
  }
  stream_operation(size_t device_address_src, void *host_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_memcpy_device_to_host;
    m_device_address_src = device_address_src;
    m_host_address_dst = host_address_dst;
    m_device_address_dst = 0;
    m_host_address_src = NULL;
    m_cnt = cnt;
    m_stream = stream;
    m_sim_mode = false;
    m_done = false;
  }
  stream_operation(size_t device_address_src, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_memcpy_device_to_device;
    m_device_address_src = device_address_src;
    m_device_address_dst = device_address_dst;
    m_host_address_src = NULL;
    m_host_address_dst = NULL;
    m_cnt = cnt;
    m_stream = stream;
    m_sim_mode = false;
    m_done = false;
  }

  bool is_kernel() const { return m_type == stream_kernel_launch; }
  bool is_mem() const {
    return m_type == stream_memcpy_host_to_device ||
           m_type == stream_memcpy_device_to_host ||
           m_type == stream_memcpy_host_to_device;
  }
  bool is_noop() const { return m_type == stream_no_op; }
  bool is_done() const { return m_done; }
  kernel_info_t *get_kernel() { return m_kernel; }
  bool do_operation(gpgpu_sim *gpu);
  void print(FILE *fp) const;
  struct CUstream_st *get_stream() { return m_stream; }
  void set_stream(CUstream_st *stream) { m_stream = stream; }
  /// yasai: host-order sequence number, assigned by stream_manager::push (0 before).
  uint64_t yasai_seq() const { return m_yasai_seq; }
  /// yasai: sets the sequence number (stream_manager::push only).
  void yasai_set_seq(uint64_t seq) { m_yasai_seq = seq; }

 private:
  uint64_t m_yasai_seq = 0;          ///< yasai: host-order sequence number
  void (*m_host_fn)(void *) = NULL;  ///< yasai: host function of a stream_host_fn operation
  void *m_host_fn_user = NULL;       ///< yasai: its argument
  struct CUstream_st *m_stream;

  bool m_done;

  stream_operation_type m_type;
  size_t m_device_address_dst;
  size_t m_device_address_src;
  void *m_host_address_dst;
  const void *m_host_address_src;
  /// yasai: owner of m_host_address_src for a copy from a pageable source (null otherwise)
  std::shared_ptr<const unsigned char[]> m_snapshot;
  size_t m_cnt;
  int m_memset_value = 0;  ///< yasai: byte value of a stream_memset operation

  const char *m_symbol;
  size_t m_offset;

  bool m_sim_mode;
  kernel_info_t *m_kernel;
  struct CUevent_st *m_event;
};
struct CUstream_st {
 public:
  CUstream_st();
  bool empty();
  bool busy();
  void synchronize();
  void push(const stream_operation &op);
  void record_next_done();
  stream_operation next();
  void cancel_front();  // front operation fails, cancle the pending status
  stream_operation &front() { return m_operations.front(); }
  void print(FILE *fp);
  unsigned get_uid() const { return m_uid; }
  /// yasai: sequence number of the front operation (the running or the next one).
  /// @param seq Output.
  /// @return False when the stream is empty.
  bool yasai_front_seq(uint64_t *seq);
  /// yasai: a stream created with cudaStreamNonBlocking is not ordered against the legacy
  /// default stream.
  bool yasai_nonblocking() const { return m_yasai_nonblocking; }
  /// yasai: sets the cudaStreamNonBlocking property (at creation).
  void yasai_set_nonblocking(bool nb) { m_yasai_nonblocking = nb; }
  // yasai: a forked child inherits the lock in whatever state the parent's threads left it.
  void reset_lock_after_fork() { pthread_mutex_init(&m_lock, NULL); }

 private:
  unsigned m_uid;
  static unsigned sm_next_stream_uid;

  std::list<stream_operation> m_operations;
  bool m_pending;  // front operation has started but not yet completed
  bool m_yasai_nonblocking = false;  // yasai: created with cudaStreamNonBlocking

  pthread_mutex_t m_lock;  // ensure only one host or gpu manipulates stream
                           // operation at one time
};

class stream_manager {
 public:
  stream_manager(gpgpu_sim *gpu, bool cuda_launch_blocking);
  bool register_finished_kernel(unsigned grid_uid);
  bool check_finished_kernel();
  stream_operation front();
  void add_stream(CUstream_st *stream);
  void destroy_stream(CUstream_st *stream);
  bool concurrent_streams_empty();
  bool empty_protected();
  bool empty();
  void print(FILE *fp);
  /// yasai: queues op on its stream and returns its sequence number. Unless the manager is in
  /// launch-blocking mode (or YASAI_STREAM_ZERO_HOST_SYNC=1 restores the former host waits),
  /// the call does not wait: the order of the legacy default stream against the blocking
  /// streams is enforced by front() on the simulation thread (see yasai_eligible).
  /// @param op Operation; its stream handle is resolved with yasai_resolve.
  /// @return Sequence number of the operation.
  uint64_t push(stream_operation op);
  /// yasai: push for a synchronous API (cudaMemcpy, cudaMemset, cudaMemcpyToSymbol, a
  /// pageable cudaMemcpyAsync to the host): returns when the operation has completed.
  /// @param op Operation.
  void push_sync(stream_operation op);
  /// yasai: waits until the operations with sequence number <= seq of stream s have
  /// completed; for the legacy default stream also those of every blocking stream.
  /// @param s Stream handle as passed by the application (resolved with yasai_resolve).
  /// @param seq Sequence bound; UINT64_MAX means everything queued so far.
  void yasai_wait_stream(CUstream_st *s, uint64_t seq);
  /// yasai: the stream an application handle denotes: NULL and cudaStreamLegacy (0x1) give
  /// NULL (the legacy default stream), cudaStreamPerThread (0x2) gives the calling thread's own
  /// blocking stream, and other handles are returned unchanged.
  /// @param s Handle.
  /// @return Stream, or NULL for the legacy default stream.
  CUstream_st *yasai_resolve(CUstream_st *s);
  void pushCudaStreamWaitEventToAllStreams(CUevent_st *e, unsigned int flags);
  bool operation(bool *sim);
  void stop_all_running_kernels();
  unsigned size() { return m_streams.size(); };
  // yasai: reinitializes the manager's and every stream's lock in a forked child.
  void reset_locks_after_fork() {
    pthread_mutex_init(&m_lock, NULL);
    m_stream_zero.reset_lock_after_fork();
    for (CUstream_st *s : m_streams) s->reset_lock_after_fork();
  }
  bool is_blocking() { return m_cuda_launch_blocking; };
  /// yasai: whether the former host-side waits of the legacy default stream are in force
  /// (launch-blocking mode or YASAI_STREAM_ZERO_HOST_SYNC=1).
  bool yasai_host_sync() const { return m_cuda_launch_blocking || m_yasai_host_sync; }

 private:
  void print_impl(FILE *fp);

  bool m_cuda_launch_blocking;
  gpgpu_sim *m_gpu;
  std::list<CUstream_st *> m_streams;
  std::map<unsigned, CUstream_st *> m_grid_id_to_stream;
  CUstream_st m_stream_zero;
  bool m_service_stream_zero;
  // yasai: whether the front operation (sequence number seq) of stream s may start: an
  // operation of the legacy stream waits for every earlier operation of the blocking streams,
  // an operation of a blocking stream waits for every earlier legacy-stream operation, and a
  // non-blocking stream is ordered only by itself. Called with m_lock held.
  bool yasai_eligible(CUstream_st *s, uint64_t seq);
  uint64_t m_yasai_next_seq = 1;  // yasai: next sequence number (under m_lock)
  size_t m_yasai_rr = 0;          // yasai: round-robin position of front()
  bool m_yasai_host_sync = false;  // yasai: YASAI_STREAM_ZERO_HOST_SYNC=1, the former waits
  std::map<pthread_t, CUstream_st *> m_yasai_per_thread;  // yasai: cudaStreamPerThread
  pthread_mutex_t m_lock;
  std::list<struct CUstream_st *>::iterator m_last_stream;
};

#endif
