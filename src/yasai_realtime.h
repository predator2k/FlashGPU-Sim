// yasai: real-time clock reads and sleeps for the simulator's own bookkeeping. A framework process
// on the fakeenv simulation backend may preload a time-dilation library (libyasai_timedilate.so)
// that interposes the libc clocks (clock_gettime, gettimeofday, time, std::chrono clocks through
// them) and sleeps (nanosleep, usleep, sleep, timed waits): the application then reads a clock
// that runs D times slower and every sleep takes D times longer. The simulator's progress timer,
// its per-kernel wall-time statistics, the polling sleeps of its host waits, the host-ahead limit's
// warning and the stall judgement's YASAI_STALL_S must keep real time, so they go through these
// functions, which enter the kernel with syscall(2) or call libc's own clock_gettime and are not
// interposed. Times the application itself sees (cudaEvent wall clock) keep the libc clock.
#ifndef YASAI_REALTIME_H
#define YASAI_REALTIME_H

#include <dlfcn.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/// Real monotonic time in seconds (CLOCK_MONOTONIC through the clock_gettime system call).
inline double yasai_real_mono_s() {
  struct timespec ts;
  if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &ts) != 0) return 0.0;
  return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}

/// Real coarse monotonic time in ns (CLOCK_MONOTONIC_COARSE), for a clock read on a hot path. The
/// clock is read through libc's own clock_gettime, looked up in libc's scope, where a preloaded
/// time-dilation library's clock_gettime is not found; libc reads the clock in user space (vDSO), so
/// the read costs what an ordinary clock_gettime costs. Without libc's symbol the system call reads it.
inline uint64_t yasai_real_coarse_ns() {
  typedef int (*clock_fn)(clockid_t, struct timespec *);
  static const clock_fn libc_clock = [] {
    void *h = dlopen("libc.so.6", RTLD_LAZY | RTLD_NOLOAD);
    return h != nullptr ? reinterpret_cast<clock_fn>(dlsym(h, "clock_gettime")) : clock_fn(nullptr);
  }();
  struct timespec ts;
  const int rc = libc_clock != nullptr
                     ? libc_clock(CLOCK_MONOTONIC_COARSE, &ts)
                     : int(syscall(SYS_clock_gettime, CLOCK_MONOTONIC_COARSE, &ts));
  if (rc != 0) return 0;
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

/// Real calendar time in whole seconds (CLOCK_REALTIME through the clock_gettime system call), the
/// value time(NULL) returns without time dilation.
inline time_t yasai_real_time() {
  struct timespec ts;
  if (syscall(SYS_clock_gettime, CLOCK_REALTIME, &ts) != 0) return (time_t)0;
  return ts.tv_sec;
}

/// Sleeps for us real microseconds (the nanosleep system call; an interrupted sleep returns early,
/// which the polling loops that call this tolerate).
/// @param us Microseconds to sleep.
inline void yasai_real_usleep(unsigned us) {
  struct timespec req;
  req.tv_sec = us / 1000000u;
  req.tv_nsec = long(us % 1000000u) * 1000L;
  syscall(SYS_nanosleep, &req, (struct timespec *)0);
}

#endif  // YASAI_REALTIME_H
