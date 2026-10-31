// yasai: real-time clock reads and sleeps for the simulator's own bookkeeping. A framework process
// on the fakeenv simulation backend may preload a time-dilation library (libyasai_timedilate.so)
// that interposes the libc clocks (clock_gettime, gettimeofday, time, std::chrono clocks through
// them) and sleeps (nanosleep, usleep, sleep, timed waits): the application then reads a clock
// that runs D times slower and every sleep takes D times longer. The simulator's progress timer,
// its per-kernel wall-time statistics and the polling sleeps of its host waits must keep real
// time, so they go through these functions, which enter the kernel with syscall(2) and are not
// interposed. Times the application itself sees (cudaEvent wall clock) keep the libc clock.
#ifndef YASAI_REALTIME_H
#define YASAI_REALTIME_H

#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/// Real monotonic time in seconds (CLOCK_MONOTONIC through the clock_gettime system call).
inline double yasai_real_mono_s() {
  struct timespec ts;
  if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &ts) != 0) return 0.0;
  return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
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
