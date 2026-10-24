// yasai: PTX for modules that a process hands to the driver at run time (see yasai_jit_image.h).
#include "yasai_jit_image.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

#include "../src/yasai_subprocess.h"

namespace yasai_jit {
namespace {

/// Magic number of a fat binary (fatbinary.h: FATBIN_MAGIC).
const uint32_t kFatbinMagic = 0xBA55ED50u;
/// Magic number of the fat binary wrapper that nvcc emits (fatbinary_section.h).
const uint32_t kFatbinWrapperMagic = 0x466243b1u;
/// Bytes of the PTX image prefix searched for ".version" and ".target".
const size_t kPtxProbeBytes = 4096;

/// Mutex around the work directory and the cache scans (loads can come from several threads).
std::mutex &jit_mutex() {
  static std::mutex m;
  return m;
}

/// True when YASAI_JIT_DEBUG is set to a nonzero value.
bool debug() {
  const char *e = getenv("YASAI_JIT_DEBUG");
  return e != NULL && atoi(e) != 0;
}

/// Work directory for extracted and written PTX (created on first use).
const std::string &work_dir() {
  static std::string dir;
  if (dir.empty()) {
    const char *w = getenv("YASAI_JIT_WORK");
    if (w != NULL && *w) {
      dir = w;
    } else {
      const char *t = getenv("TMPDIR");
      dir = std::string(t != NULL && *t ? t : "/tmp") + "/yasai-jit-" + std::to_string(getpid());
    }
    yasai_system(("mkdir -p '" + dir + "'").c_str());
  }
  return dir;
}

/// FNV-1a 64-bit hash of a byte range, as 16 hex digits.
///   p  first byte
///   n  number of bytes
std::string hash_hex(const unsigned char *p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
  return buf;
}

/// Writes n bytes to path (overwrites).
///   path  output file
///   p     data
///   n     number of bytes
bool write_file(const std::string &path, const void *p, size_t n) {
  std::ofstream out(path.c_str(), std::ios::binary);
  out.write((const char *)p, (std::streamsize)n);
  return out.good();
}

/// Reads a whole file into s; false when it cannot be opened.
///   path  input file
///   s     receives the contents
bool read_file(const std::string &path, std::string *s) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in.good()) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  *s = ss.str();
  return true;
}

/// Numeric architecture of an "sm_XX[suffix]" string (0 when there is none).
///   arch  for example "sm_120a"
unsigned arch_number(const std::string &arch) {
  if (arch.compare(0, 3, "sm_") != 0) return 0;
  return (unsigned)atoi(arch.c_str() + 3);
}

/// Selects, from PTX files extracted by cuobjdump, the files of the highest architecture that
/// is at most max_capability (or of the highest one when none is), preferring suffixed
/// architectures (sm_90a over sm_90) as the main loader does.
///   files           extracted PTX file paths
///   max_capability  highest architecture to select (0: the highest available)
std::vector<std::string> select_arch(const std::vector<std::string> &files,
                                     unsigned max_capability) {
  std::map<unsigned, std::vector<std::string>> by_arch;
  for (const std::string &f : files) by_arch[arch_number(arch_of_ptx_file(f))].push_back(f);
  if (by_arch.empty()) return {};
  unsigned chosen = by_arch.rbegin()->first;
  if (max_capability != 0) {
    for (auto it = by_arch.rbegin(); it != by_arch.rend(); ++it)
      if (it->first <= max_capability) {
        chosen = it->first;
        break;
      }
  }
  std::vector<std::string> sel = by_arch[chosen], suffixed;
  for (const std::string &f : sel) {
    const std::string a = arch_of_ptx_file(f);
    if (!a.empty() && isalpha((unsigned char)a.back())) suffixed.push_back(f);
  }
  return suffixed.empty() ? sel : suffixed;
}

/// Runs `cuobjdump -xptx all` on a file in a fresh directory and returns the PTX files.
///   file  library, cubin or fat binary file
///   dir   output directory (created)
std::vector<std::string> extract_ptx(const std::string &file, const std::string &dir) {
  const std::string cmd = "mkdir -p '" + dir + "' && cd '" + dir +
                          "' && $CUDA_INSTALL_PATH/bin/cuobjdump -xptx all '" + file +
                          "' > /dev/null 2>&1";
  yasai_system(cmd.c_str());
  std::vector<std::string> out;
  if (DIR *d = opendir(dir.c_str())) {
    while (struct dirent *e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 4 && n.compare(n.size() - 4, 4, ".ptx") == 0) out.push_back(dir + "/" + n);
    }
    closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Size in bytes of an ELF64 image: the end of its last section, program or section header.
///   p  image start (ELF magic checked by the caller)
size_t elf_size(const unsigned char *p) {
  uint64_t phoff, shoff;
  uint16_t phentsize, phnum, shentsize, shnum;
  memcpy(&phoff, p + 0x20, 8);
  memcpy(&shoff, p + 0x28, 8);
  memcpy(&phentsize, p + 0x36, 2);
  memcpy(&phnum, p + 0x38, 2);
  memcpy(&shentsize, p + 0x3A, 2);
  memcpy(&shnum, p + 0x3C, 2);
  size_t end = std::max<size_t>(phoff + (size_t)phentsize * phnum, shoff + (size_t)shentsize * shnum);
  for (unsigned i = 0; i < shnum; ++i) {
    const unsigned char *sh = p + shoff + (size_t)i * shentsize;
    uint32_t type;
    uint64_t off, size;
    memcpy(&type, sh + 4, 4);
    memcpy(&off, sh + 0x18, 8);
    memcpy(&size, sh + 0x20, 8);
    if (type != 8 /* SHT_NOBITS */) end = std::max<size_t>(end, off + size);
  }
  return end;
}

/// Kernel names of an ELF64 cubin: the names of its ".text.<kernel>" sections.
///   p  image start (ELF magic checked by the caller)
std::vector<std::string> elf_kernels(const unsigned char *p) {
  uint64_t shoff;
  uint16_t shentsize, shnum, shstrndx;
  memcpy(&shoff, p + 0x28, 8);
  memcpy(&shentsize, p + 0x3A, 2);
  memcpy(&shnum, p + 0x3C, 2);
  memcpy(&shstrndx, p + 0x3E, 2);
  std::vector<std::string> out;
  if (shstrndx >= shnum) return out;
  uint64_t stroff;
  memcpy(&stroff, p + shoff + (size_t)shstrndx * shentsize + 0x18, 8);
  for (unsigned i = 0; i < shnum; ++i) {
    uint32_t name;
    memcpy(&name, p + shoff + (size_t)i * shentsize, 4);
    const char *s = (const char *)p + stroff + name;
    if (strncmp(s, ".text.", 6) == 0) out.push_back(s + 6);
  }
  return out;
}

/// Cache directories searched for cubin images (see YASAI_JIT_PTX_DIRS).
std::vector<std::string> cache_dirs() {
  std::vector<std::string> dirs;
  std::string list;
  if (const char *e = getenv("YASAI_JIT_PTX_DIRS")) list = e;
  if (list.empty()) {
    const char *home = getenv("HOME");
    const std::string h = home != NULL ? home : "";
    const char *tc = getenv("TRITON_CACHE_DIR");
    list = (tc != NULL && *tc ? std::string(tc) : h + "/.triton/cache") + ":" + h +
           "/.cache/flashinfer:" + h + "/.deep_gemm";
  }
  size_t start = 0;
  while (start <= list.size()) {
    const size_t end = std::min(list.find(':', start), list.size());
    if (end > start) dirs.push_back(list.substr(start, end - start));
    start = end + 1;
  }
  return dirs;
}

/// Appends every regular file below dir whose name ends with suffix.
///   dir     directory searched recursively
///   suffix  for example ".cubin"
///   out     receives the paths
void find_files(const std::string &dir, const std::string &suffix, std::vector<std::string> *out) {
  DIR *d = opendir(dir.c_str());
  if (d == NULL) return;
  while (struct dirent *e = readdir(d)) {
    const std::string n = e->d_name;
    if (n == "." || n == "..") continue;
    const std::string p = dir + "/" + n;
    struct stat st;
    if (stat(p.c_str(), &st) != 0) continue;
    if (S_ISDIR(st.st_mode)) {
      find_files(p, suffix, out);
    } else if (n.size() > suffix.size() &&
               n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
      out->push_back(p);
    }
  }
  closedir(d);
}

/// True when the PTX text defines a kernel (".entry") of the given name.
///   text  PTX text
///   name  kernel name
bool defines_entry(const std::string &text, const std::string &name) {
  size_t pos = 0;
  while ((pos = text.find(name, pos)) != std::string::npos) {
    const size_t end = pos + name.size();
    const bool word_end = end >= text.size() || !(isalnum((unsigned char)text[end]) || text[end] == '_' || text[end] == '$');
    const size_t ls = text.rfind('\n', pos);
    const std::string head = text.substr(ls == std::string::npos ? 0 : ls + 1,
                                         pos - (ls == std::string::npos ? 0 : ls + 1));
    if (word_end && head.find(".entry") != std::string::npos) return true;
    pos = end;
  }
  return false;
}

/// PTX for an ELF cubin from the JIT caches, or "".
///   p     image start
///   size  image size in bytes
///   how   receives the description of the match
std::string ptx_for_cubin(const unsigned char *p, size_t size, std::string *how) {
  const std::vector<std::string> dirs = cache_dirs();
  // 1. A cached cubin with the same bytes, and the .ptx next to it.
  const std::string want = hash_hex(p, size);
  for (const std::string &d : dirs) {
    std::vector<std::string> cubins;
    find_files(d, ".cubin", &cubins);
    for (const std::string &c : cubins) {
      struct stat st;
      if (stat(c.c_str(), &st) != 0 || (size_t)st.st_size != size) continue;
      std::string bytes;
      if (!read_file(c, &bytes) || hash_hex((const unsigned char *)bytes.data(), bytes.size()) != want)
        continue;
      const std::string ptx = c.substr(0, c.size() - 6) + ".ptx";
      struct stat pst;
      if (stat(ptx.c_str(), &pst) == 0) {
        *how = "cubin " + want + " matches " + c + ", PTX next to it";
        return ptx;
      }
    }
  }
  // 2. A cached PTX file that defines every kernel of the cubin.
  const std::vector<std::string> kernels = elf_kernels(p);
  if (kernels.empty()) return "";
  for (const std::string &d : dirs) {
    std::vector<std::string> ptxs;
    find_files(d, ".ptx", &ptxs);
    for (const std::string &f : ptxs) {
      std::string text;
      if (!read_file(f, &text)) continue;
      bool all = true;
      for (const std::string &k : kernels)
        if (!defines_entry(text, k)) {
          all = false;
          break;
        }
      if (all) {
        *how = "cubin " + want + " kernels (" + kernels[0] + (kernels.size() > 1 ? ", ..." : "") +
               ") defined in " + f;
        return f;
      }
    }
  }
  *how = "cubin " + want + " (" + kernels[0] + ") not found in the JIT caches";
  return "";
}

}  // namespace

std::string arch_of_ptx_file(const std::string &ptx_file) {
  const size_t slash = ptx_file.rfind('/');
  const std::string base = slash == std::string::npos ? ptx_file : ptx_file.substr(slash + 1);
  const size_t sm = base.rfind(".sm_");
  if (sm == std::string::npos) return "";
  const size_t end = base.find('.', sm + 1);
  return base.substr(sm + 1, end == std::string::npos ? std::string::npos : end - sm - 1);
}

std::vector<std::string> ptx_files_of_library(const std::string &path, unsigned max_capability) {
  std::lock_guard<std::mutex> lock(jit_mutex());
  const std::string dir = work_dir() + "/lib-" +
                          hash_hex((const unsigned char *)path.data(), path.size());
  return select_arch(extract_ptx(path, dir), max_capability);
}

std::string ptx_for_image(const void *image, unsigned max_capability, std::string *how) {
  std::lock_guard<std::mutex> lock(jit_mutex());
  std::string ignored;
  if (how == NULL) how = &ignored;
  const unsigned char *p = (const unsigned char *)image;
  if (p == NULL) {
    *how = "null image";
    return "";
  }
  uint32_t magic;
  memcpy(&magic, p, 4);
  if (magic == kFatbinWrapperMagic) {
    // struct { int magic; int version; const unsigned long long *data; void *filename; }
    const unsigned char *data;
    memcpy(&data, p + 8, sizeof(data));
    p = data;
    memcpy(&magic, p, 4);
  }
  std::string result;
  if (magic == kFatbinMagic) {
    // fat binary header: magic (4), version (2), header size (2), payload size (8)
    uint16_t header_size;
    uint64_t fat_size;
    memcpy(&header_size, p + 6, 2);
    memcpy(&fat_size, p + 8, 8);
    const size_t n = header_size + fat_size;
    const std::string h = hash_hex(p, n);
    const std::string file = work_dir() + "/" + h + ".fatbin";
    write_file(file, p, n);
    const std::vector<std::string> sel = select_arch(extract_ptx(file, work_dir() + "/" + h), max_capability);
    if (!sel.empty()) {
      *how = "fat binary " + h + ", embedded PTX " + sel[0];
      result = sel[0];
    } else {
      *how = "fat binary " + h + " carries no PTX";
    }
  } else if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
    result = ptx_for_cubin(p, elf_size(p), how);
  } else {
    // PTX text: ".version" and ".target" within the first kPtxProbeBytes bytes.
    const size_t n = strnlen((const char *)p, kPtxProbeBytes);
    const std::string head((const char *)p, n);
    if (head.find(".version") != std::string::npos && head.find(".target") != std::string::npos) {
      const size_t len = strlen((const char *)p);
      const std::string h = hash_hex(p, len);
      result = work_dir() + "/" + h + ".ptx";
      write_file(result, p, len);
      *how = "PTX image " + h;
    } else {
      *how = "unknown image format";
    }
  }
  if (debug()) printf("yasai_jit: %s\n", how->c_str());
  return result;
}

}  // namespace yasai_jit
