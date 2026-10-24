// yasai: fakeenv CUDA recorder of the functional simulation backend.
//
// When YASAI_FAKEENV_OUT is set, the runtime API writes the records the fakeenv CUPTI recorder
// writes on a real GPU (sim/tools/fakeenv/recorders/cupti), so that fakeenv_to_ir.py converts a
// simulated run like a recorded one: one JSON line per record in
// $YASAI_FAKEENV_OUT/<host>-r<rank>-p<pid>-sim/cuda.jsonl, starting with a header. Records:
//   api            launch, copy, memset, event and sync calls (domain "runtime")
//   kernel/memcpy/memset  the device operation of a call, with the call's correlation id
//   stream_created the CUstream pointer to stream id map (NCCL's profiler names streams by
//                  pointer)
//   event_record / sync  cross-stream dependencies (cudaEventRecord, cudaStreamWaitEvent) and
//                  host waits, with the CUPTI synchronization types
// Times are CLOCK_MONOTONIC ns of the host call: the functional mode has no device time, so a
// device operation gets the start and end of its launch call (the header says so). Stream ids:
// the legacy default stream is 7 (as CUPTI numbers it); created streams get 13, 14, ...
// A forked child (zygote ranks) opens its own file at its first record.
#ifndef YASAI_FAKEENV_REC_H
#define YASAI_FAKEENV_REC_H

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <map>
#include <mutex>
#include <string>

namespace yasai_rec {

/// CUPTI synchronization types used in "sync" records.
enum SyncType { kEventSync = 1, kStreamWaitEvent = 2, kStreamSync = 3, kContextSync = 4 };

/// CLOCK_MONOTONIC in ns.
inline uint64_t mono_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

/// CLOCK_REALTIME in ns.
inline uint64_t real_ns() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

/// JSON string escaping of a kernel name (quotes and backslashes; names are ASCII).
inline std::string esc(const char *s) {
  std::string o;
  for (; s && *s; ++s) {
    if (*s == '"' || *s == '\\') o += '\\';
    o += *s;
  }
  return o;
}

/// Writer of one process's cuda.jsonl.
class Recorder {
 public:
  /// The process-wide recorder.
  static Recorder &get() {
    static Recorder r;
    return r;
  }

  /// Whether recording is on (YASAI_FAKEENV_OUT set).
  bool on() const { return out_ != nullptr && *out_; }

  /// Next correlation id.
  uint32_t corr() {
    std::lock_guard<std::mutex> l(mu_);
    return ++corr_;
  }

  /// Stream id of a stream (7 for the legacy default stream); emits stream_created once.
  uint32_t stream_id(const void *stream) {
    std::lock_guard<std::mutex> l(mu_);
    return stream_id_locked(stream);
  }

  /// Writes one record (printf-style JSON object body without braces).
  void write(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
    if (!on()) return;
    std::lock_guard<std::mutex> l(mu_);
    if (!open_locked()) return;
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(f_, "{%s}\n", buf);
    fflush(f_);
  }

  /// Writes an api record.
  void api(const char *name, uint64_t start, uint64_t end, uint32_t corr, int ret) {
    write("\"rec\":\"api\",\"domain\":\"runtime\",\"name\":\"%s\",\"start\":%llu,\"end\":%llu,"
          "\"tid\":%ld,\"corr\":%u,\"ret\":%d",
          name, (unsigned long long)start, (unsigned long long)end, (long)syscall(SYS_gettid),
          corr, ret);
  }

  /// Sync id of the next record of an event (and remembers it as the event's last one).
  uint64_t event_record_id(uint64_t event) {
    std::lock_guard<std::mutex> l(mu_);
    return last_sync_[event] = ++sync_seq_;
  }

  /// Sync id of an event's last record (0 when it was never recorded).
  uint64_t event_last_id(uint64_t event) {
    std::lock_guard<std::mutex> l(mu_);
    auto it = last_sync_.find(event);
    return it == last_sync_.end() ? 0 : it->second;
  }

 private:
  Recorder() : out_(getenv("YASAI_FAKEENV_OUT")) {}

  uint32_t stream_id_locked(const void *stream) {
    if (stream == nullptr) return 7;
    auto it = ids_.find(stream);
    if (it != ids_.end()) return it->second;
    const uint32_t id = next_id_++;
    ids_[stream] = id;
    if (open_locked())
      fprintf(f_, "{\"rec\":\"stream_created\",\"t\":%llu,\"cuctx\":\"0x1\",\"custream\":\"%p\","
                  "\"stream\":%u}\n",
              (unsigned long long)mono_ns(), stream, id);
    return id;
  }

  /// Opens (or, in a forked child, reopens) the file and writes the header; caller holds mu_.
  bool open_locked() {
    const pid_t pid = getpid();
    if (f_ != nullptr && pid == pid_) return true;
    if (f_ != nullptr) fclose(f_);
    f_ = nullptr;
    pid_ = pid;
    corr_ = 0;
    char host[256] = "host";
    gethostname(host, sizeof host - 1);
    long rank = -1;
    for (const char *k : {"YASAI_FAKEENV_RANK", "RANK", "OMPI_COMM_WORLD_RANK"}) {
      if (const char *v = getenv(k)) {
        rank = atol(v);
        break;
      }
    }
    std::string dir = std::string(out_) + "/" + host + "-r" + std::to_string(rank) + "-p" +
                      std::to_string(pid) + "-sim";
    mkdir(out_, 0777);
    if (mkdir(dir.c_str(), 0777) != 0 && errno != EEXIST) return false;
    f_ = fopen((dir + "/cuda.jsonl").c_str(), "a");
    if (f_ == nullptr) return false;
    fprintf(f_, "{\"rec\":\"header\",\"layer\":\"cuda\",\"host\":\"%s\",\"pid\":%d,\"rank\":%ld,"
                "\"mono_ns\":%llu,\"real_ns\":%llu,\"source\":\"flashgpu-sim functional; device "
                "operations carry the host time of their launch call\"}\n",
            host, (int)pid, rank, (unsigned long long)mono_ns(), (unsigned long long)real_ns());
    fprintf(f_, "{\"rec\":\"init\",\"t\":%llu,\"clock\":\"host_monotonic_ns\"}\n",
            (unsigned long long)mono_ns());
    // Streams created before a fork keep their ids in the child; announce them again.
    for (const auto &kv : ids_)
      fprintf(f_, "{\"rec\":\"stream_created\",\"t\":%llu,\"cuctx\":\"0x1\",\"custream\":\"%p\","
                  "\"stream\":%u}\n",
              (unsigned long long)mono_ns(), kv.first, kv.second);
    return true;
  }

  const char *out_;                        ///< YASAI_FAKEENV_OUT.
  FILE *f_ = nullptr;                      ///< cuda.jsonl of this process.
  pid_t pid_ = 0;                          ///< Process that opened f_.
  std::mutex mu_;                          ///< Guards everything below.
  uint32_t corr_ = 0;                      ///< Last correlation id.
  uint32_t next_id_ = 13;                  ///< Next stream id.
  std::map<const void *, uint32_t> ids_;   ///< Stream pointer to id.
  uint64_t sync_seq_ = 0;                  ///< Last event-record sync id.
  std::map<uint64_t, uint64_t> last_sync_; ///< Event uid to the sync id of its last record.
};

/// CUPTI copy kind of a cudaMemcpyKind value.
inline unsigned copy_kind(int kind) {
  switch (kind) {
    case 1: return 1;   // HostToDevice -> HTOD
    case 2: return 2;   // DeviceToHost -> DTOH
    case 3: return 8;   // DeviceToDevice -> DTOD
    case 0: return 9;   // HostToHost -> HTOH
    default: return 8;  // Default (unified addressing): device to device
  }
}

}  // namespace yasai_rec

#endif  // YASAI_FAKEENV_REC_H
