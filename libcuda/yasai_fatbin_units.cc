// yasai: translation units of a library's fat binaries (see yasai_fatbin_units.h).
#include "yasai_fatbin_units.h"

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "../src/cuda-sim/yasai_module_scope.h"

namespace yasai_fatbin {
namespace {

/// Magic number of a fat binary header in .nv_fatbin.
const uint32_t kFatbinMagic = 0xBA55ED50u;
/// Entry kind of a PTX entry in a fat binary.
const uint16_t kEntryKindPtx = 1;

/// One fat binary of an ELF file's .nv_fatbin section.
struct blob_t {
  uint64_t vaddr = 0;       ///< virtual address of its header (as linked)
  unsigned first_ptx = 0;   ///< cuobjdump number of its first PTX entry
  unsigned n_ptx = 0;       ///< number of its PTX entries
  uint64_t file_off = 0;    ///< file offset of its header
  uint64_t bytes = 0;       ///< header plus payload size
};

/// The fat binaries of one ELF file.
struct elf_info_t {
  bool ok = false;               ///< whether .nv_fatbin was read
  bool exec = false;             ///< ET_EXEC (addresses are absolute) rather than ET_DYN
  std::vector<blob_t> blobs;     ///< in section order
};

/// Process-wide state.
struct state_t {
  std::mutex mu;                                 ///< guards the maps
  std::map<std::string, elf_info_t> elf;         ///< ELF path -> its fat binaries
  std::set<std::string> noted;                   ///< ELF files noted with note_elf
  std::map<std::string, std::pair<std::string, unsigned>> listed;  ///< extracted PTX file
                                                 ///< (canonical path) -> (ELF file, number)
};

/// The process-wide state.
/// @return The state.
state_t &st() {
  static state_t s;
  return s;
}

/// Base name of a path.
/// @param p Path.
/// @return The part after the last '/'.
std::string base_name(const std::string &p) {
  const size_t s = p.rfind('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

/// Reads the fat binaries of an ELF file's .nv_fatbin section.
/// @param path ELF file.
/// @return The section's fat binaries (ok false when the file or section cannot be read).
elf_info_t read_elf(const std::string &path) {
  elf_info_t r;
  std::ifstream in(path, std::ios::binary);
  if (!in) return r;
  unsigned char eh[64];
  if (!in.read((char *)eh, sizeof eh) || memcmp(eh, "\x7f" "ELF", 4) != 0 || eh[4] != 2) return r;
  uint16_t type, shentsize, shnum, shstrndx;
  uint64_t shoff;
  memcpy(&type, eh + 16, 2);
  memcpy(&shoff, eh + 0x28, 8);
  memcpy(&shentsize, eh + 0x3A, 2);
  memcpy(&shnum, eh + 0x3C, 2);
  memcpy(&shstrndx, eh + 0x3E, 2);
  r.exec = type == 2;
  if (shentsize < 64 || shnum == 0 || shstrndx >= shnum) return r;
  std::vector<unsigned char> sh(size_t(shentsize) * shnum);
  in.seekg(std::streamoff(shoff));
  if (!in.read((char *)sh.data(), std::streamsize(sh.size()))) return r;
  auto field64 = [&](unsigned i, unsigned off) {
    uint64_t v;
    memcpy(&v, sh.data() + size_t(i) * shentsize + off, 8);
    return v;
  };
  auto name_off = [&](unsigned i) {
    uint32_t v;
    memcpy(&v, sh.data() + size_t(i) * shentsize, 4);
    return v;
  };
  std::vector<char> names(field64(shstrndx, 0x20));
  in.seekg(std::streamoff(field64(shstrndx, 0x18)));
  if (!in.read(names.data(), std::streamsize(names.size()))) return r;
  for (unsigned i = 0; i < shnum; ++i) {
    const uint32_t n = name_off(i);
    if (n >= names.size() || strcmp(names.data() + n, ".nv_fatbin") != 0) continue;
    const uint64_t addr = field64(i, 0x10), off = field64(i, 0x18), size = field64(i, 0x20);
    std::vector<unsigned char> d(size);
    in.seekg(std::streamoff(off));
    if (!in.read((char *)d.data(), std::streamsize(size))) return r;
    unsigned next_ptx = 1;
    size_t p = 0;
    while (p + 16 <= d.size()) {
      uint32_t magic;
      uint16_t hsz;
      uint64_t fsz;
      memcpy(&magic, d.data() + p, 4);
      if (magic != kFatbinMagic) {  // padding between fat binaries
        p += 8;
        continue;
      }
      memcpy(&hsz, d.data() + p + 6, 2);
      memcpy(&fsz, d.data() + p + 8, 8);
      blob_t b;
      b.vaddr = addr + p;
      b.first_ptx = next_ptx;
      b.file_off = off + p;
      b.bytes = hsz + fsz;
      size_t e = p + hsz;
      const size_t end = p + hsz + fsz;
      while (e + 16 <= end && end <= d.size()) {
        uint16_t kind;
        uint32_t ehsz;
        uint64_t psz;
        memcpy(&kind, d.data() + e, 2);
        memcpy(&ehsz, d.data() + e + 4, 4);
        memcpy(&psz, d.data() + e + 8, 8);
        if (kind == kEntryKindPtx) ++b.n_ptx;
        if (ehsz == 0) break;
        e += ehsz + psz;
      }
      next_ptx += b.n_ptx;
      r.blobs.push_back(b);
      p = end;
    }
    r.ok = true;
    break;
  }
  return r;
}

/// The fat binaries of an ELF file, read once; the caller holds st().mu.
/// @param path ELF file.
/// @return Its fat binaries.
const elf_info_t &elf_locked(const std::string &path) {
  auto it = st().elf.find(path);
  if (it == st().elf.end()) it = st().elf.emplace(path, read_elf(path)).first;
  return it->second;
}

/// Canonical path (realpath, or the path itself).
/// @param p Path.
/// @return The canonical path.
std::string canonical(const std::string &p) {
  char buf[PATH_MAX];
  return realpath(p.c_str(), buf) ? std::string(buf) : p;
}

/// Unit string of fat binary k of an ELF file.
/// @param elf_path ELF file.
/// @param k Fat binary number (0-based, section order).
/// @return "<library key>#<k>".
std::string unit_string(const std::string &elf_path, size_t k) {
  return yasai_scope::library_unit(elf_path) + "#" + std::to_string(k);
}

}  // namespace

std::string unit_of_ptx_number(const std::string &elf_path, unsigned ptx_number) {
  const std::string path = canonical(elf_path);
  std::lock_guard<std::mutex> lock(st().mu);
  const elf_info_t &e = elf_locked(path);
  if (!e.ok) return "";
  for (size_t k = 0; k < e.blobs.size(); ++k) {
    const blob_t &b = e.blobs[k];
    if (ptx_number >= b.first_ptx && ptx_number < b.first_ptx + b.n_ptx) return unit_string(path, k);
  }
  return "";
}

std::string unit_of_ptx_file(const std::string &ptx_file, const std::string &elf_path) {
  {
    std::pair<std::string, unsigned> where;
    bool found = false;
    {
      std::lock_guard<std::mutex> lock(st().mu);
      auto it = st().listed.find(canonical(ptx_file));
      if (it != st().listed.end()) {
        where = it->second;
        found = true;
      }
    }
    if (found) return unit_of_ptx_number(where.first, where.second);
  }
  // "<name>.<n>.sm_XX.ptx": the number is the dot-separated field before ".sm_"
  const std::string base = base_name(ptx_file);
  const size_t sm = base.rfind(".sm_");
  if (sm == std::string::npos || sm == 0) return "";
  const size_t dot = base.rfind('.', sm - 1);
  if (dot == std::string::npos) return "";
  const std::string num = base.substr(dot + 1, sm - dot - 1);
  if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos) return "";
  if (!elf_path.empty()) return unit_of_ptx_number(elf_path, unsigned(atoi(num.c_str())));
  const std::string key = yasai_scope::library_unit(base.substr(0, dot));
  std::vector<std::string> cands;
  {
    std::lock_guard<std::mutex> lock(st().mu);
    cands.assign(st().noted.begin(), st().noted.end());
  }
  char exe[PATH_MAX];
  const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (n > 0) {
    exe[n] = '\0';
    cands.push_back(exe);
  }
  for (const std::string &c : cands)
    if (yasai_scope::library_unit(c) == key) return unit_of_ptx_number(c, unsigned(atoi(num.c_str())));
  return "";
}

std::string unit_of_fatbin(const void *fat_cubin) {
  if (fat_cubin == NULL) return "";
  // wrapper: { int magic; int version; const unsigned long long *data; void *filename }
  const unsigned char *data = NULL;
  memcpy(&data, (const unsigned char *)fat_cubin + 8, sizeof data);
  Dl_info info;
  if (data == NULL || !dladdr(data, &info) || info.dli_fname == NULL || info.dli_fname[0] == '\0')
    return "";
  std::string path = info.dli_fname;
  if (path.find('/') == std::string::npos) {  // the program itself is reported by its argv[0]
    char exe[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n > 0) {
      exe[n] = '\0';
      path = exe;
    }
  }
  path = canonical(path);
  std::lock_guard<std::mutex> lock(st().mu);
  const elf_info_t &e = elf_locked(path);
  if (!e.ok) return "";
  const uint64_t vaddr = e.exec ? uint64_t(data) : uint64_t(data) - uint64_t(info.dli_fbase);
  for (size_t k = 0; k < e.blobs.size(); ++k)
    if (e.blobs[k].vaddr == vaddr) return unit_string(path, k);
  return "";
}

void note_elf(const std::string &elf_path) {
  const std::string path = canonical(elf_path);
  std::lock_guard<std::mutex> lock(st().mu);
  st().noted.insert(path);
}

std::vector<std::vector<std::string>> settle_listing(
    const std::vector<std::string> &elf_paths, const std::vector<std::vector<std::string>> &names,
    const std::string &dir, int (*run)(const char *)) {
  std::map<std::string, int> count;  // listed name -> times listed in dir
  for (const auto &v : names)
    for (const std::string &n : v) ++count[n];
  std::vector<std::vector<std::string>> out = names;
  const std::string split_dir = dir + "/yasai_fatbin_split";
  for (size_t e = 0; e < elf_paths.size() && e < names.size(); ++e) {
    const std::string elf = canonical(elf_paths[e]);
    elf_info_t info;
    {
      std::lock_guard<std::mutex> lock(st().mu);
      info = elf_locked(elf);
    }
    const std::string key = yasai_scope::library_unit(elf);
    std::map<size_t, std::string> blob_dir;  // fat binary number -> its extraction directory
    for (size_t i = 0; i < names[e].size(); ++i) {
      const unsigned number = unsigned(i + 1);
      const std::string &name = names[e][i];
      if (count[name] > 1 && info.ok) {
        // The name was overwritten by another listed file: extract this entry from its own
        // fat binary and store it under a unique numbered name.
        size_t k = info.blobs.size();
        for (size_t b = 0; b < info.blobs.size(); ++b)
          if (number >= info.blobs[b].first_ptx &&
              number < info.blobs[b].first_ptx + info.blobs[b].n_ptx)
            k = b;
        const size_t sm = name.rfind(".sm_");
        if (k < info.blobs.size() && sm != std::string::npos) {
          if (!blob_dir.count(k)) {
            const std::string bd = split_dir + "/" + key + "." + std::to_string(k);
            mkdir(split_dir.c_str(), 0755);
            mkdir(bd.c_str(), 0755);
            std::vector<char> bytes(info.blobs[k].bytes);
            std::ifstream in(elf, std::ios::binary);
            in.seekg(std::streamoff(info.blobs[k].file_off));
            in.read(bytes.data(), std::streamsize(bytes.size()));
            const std::string fb = bd + "/blob.fatbin";
            std::ofstream(fb, std::ios::binary).write(bytes.data(), std::streamsize(bytes.size()));
            const std::string cmd = "cd '" + bd +
                                    "' && $CUDA_INSTALL_PATH/bin/cuobjdump -xptx all blob.fatbin "
                                    "> /dev/null 2>&1";
            run(cmd.c_str());
            blob_dir[k] = bd;
          }
          const std::string renamed = key + "." + std::to_string(number) + name.substr(sm);
          if (rename((blob_dir[k] + "/" + name).c_str(), (dir + "/" + renamed).c_str()) == 0) {
            printf("GPGPU-Sim PTX: %s is listed %d times; entry %u of %s is %s\n", name.c_str(),
                   count[name], number, elf.c_str(), renamed.c_str());
            out[e][i] = renamed;
          }
        }
      }
      std::lock_guard<std::mutex> lock(st().mu);
      st().listed[canonical(dir + "/" + out[e][i])] = std::make_pair(elf, number);
    }
  }
  return out;
}

}  // namespace yasai_fatbin
