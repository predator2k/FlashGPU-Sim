// yasai: shell commands of the simulator (ptxas, cuobjdump, md5sum, ldd, rm) run without
// LD_PRELOAD. A program that preloads libraries linked against the simulator's libcudart (the
// fakeenv cuBLAS replacement, NCCL) would otherwise start the simulator again in every shell
// it spawns, and that copy runs the same commands, without end.
#ifndef YASAI_SUBPROCESS_H
#define YASAI_SUBPROCESS_H

#include <stdio.h>
#include <stdlib.h>

#include <mutex>
#include <string>

/// Mutex around the LD_PRELOAD swap (commands run from several threads during lazy loading).
inline std::mutex &yasai_subprocess_mutex() {
  static std::mutex m;
  return m;
}

/// Runs cmd with system(3), LD_PRELOAD removed for the child.
inline int yasai_system(const char *cmd) {
  std::lock_guard<std::mutex> lock(yasai_subprocess_mutex());
  const char *pre = getenv("LD_PRELOAD");
  const std::string saved = pre ? pre : "";
  if (pre) unsetenv("LD_PRELOAD");
  const int r = system(cmd);
  if (pre) setenv("LD_PRELOAD", saved.c_str(), 1);
  return r;
}

/// popen(3) with LD_PRELOAD removed for the child.
inline FILE *yasai_popen(const char *cmd, const char *mode) {
  std::lock_guard<std::mutex> lock(yasai_subprocess_mutex());
  const char *pre = getenv("LD_PRELOAD");
  const std::string saved = pre ? pre : "";
  if (pre) unsetenv("LD_PRELOAD");
  FILE *f = popen(cmd, mode);
  if (pre) setenv("LD_PRELOAD", saved.c_str(), 1);
  return f;
}

#endif  // YASAI_SUBPROCESS_H
