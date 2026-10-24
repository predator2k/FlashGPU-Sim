// yasai: per-module scope for PTX symbols without linkage (see yasai_module_scope.h).
#include "yasai_module_scope.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace yasai_scope {
namespace {

/// Directory, relative to the working directory, that holds the renamed copies.
const char *const kScopeDir = "yasai_module_scope";

/// State shared by all calls: the copy made for each scanned path and, for every renamed name,
/// the (file base name, renamed symbol) pairs.
struct Registry {
  std::mutex mu;                                    ///< guards the maps
  std::map<std::string, std::string> scoped_path;   ///< scanned path -> file to parse
  std::map<std::string, std::string> base_owner;    ///< copy base name -> the path it came from
  std::set<std::string> defined;                    ///< names some module defines with linkage
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> renamed;  ///< name ->
                                                    ///< (file base name, renamed symbol)
  std::map<std::string, std::string> entry_unit;    ///< .entry name -> unit that defined it first
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> entry_renamed;  ///<
                                                    ///< .entry name -> (unit, renamed symbol)
  std::map<std::string, std::map<std::string, std::string>> path_entries;  ///< scanned path ->
                                                    ///< (.entry name -> renamed symbol)
  std::map<std::string, std::string> entry_source;  ///< renamed .entry symbol -> source name
};

/// The process-wide registry.
/// @return The registry.
Registry &registry() {
  static Registry r;
  return r;
}

/// True for a character that can continue a PTX identifier.
/// @param c Character.
/// @return Whether c is a letter, digit, '_' or '$'.
bool ident_char(char c) { return isalnum((unsigned char)c) || c == '_' || c == '$'; }

/// True for a character that can start a PTX identifier.
/// @param c Character.
/// @return Whether c is a letter, '_' or '$'.
bool ident_start(char c) { return isalpha((unsigned char)c) || c == '_' || c == '$'; }

/// Base name of a path (the part after the last '/').
/// @param path Path.
/// @return The base name.
std::string base_name(const std::string &path) {
  const size_t s = path.rfind('/');
  return s == std::string::npos ? path : path.substr(s + 1);
}

/// Library key of a file name: for an extracted PTX file ("libfoo.so.1.sm_120.ptx" or
/// "libfoo.1.sm_120a.ptx") the part before ".<n>.sm_", and for a library ("libfoo.so.3") the base
/// name; in both, ".so" and what follows it are dropped ("libfoo").
/// @param name File base name.
/// @return The key.
std::string library_key(const std::string &name) {
  std::string k = name;
  const size_t sm = k.find("sm_");
  if (sm != std::string::npos && sm >= 2) {
    const size_t cut = k.rfind('.', sm - 2);
    if (cut != std::string::npos) k = k.substr(0, cut);
  }
  const size_t so = k.find(".so");
  if (so != std::string::npos && (so + 3 == k.size() || k[so + 3] == '.')) k = k.substr(0, so);
  return k;
}

/// Unit of a PTX file for kernel names: its library key without an architecture part, so all
/// architecture variants of one binary are one unit ("libfoo.1.sm_120.ptx" and
/// "libfoo.2.sm_120f.ptx" give "libfoo"; "app.sm_90a.ptx" and "app.sm_90.ptx" give "app").
/// @param base File base name.
/// @return The unit.
std::string file_unit(const std::string &base) {
  std::string k = library_key(base);
  for (const char *arch : {".sm_", ".compute_"}) {
    const size_t a = k.find(arch);
    if (a != std::string::npos) k = k.substr(0, a);
  }
  return k;
}

/// Suffix that makes a module-local name unique to one file: "__ym" and the 32-bit FNV-1a hash
/// of the file's base name in hex (deterministic, so ptxas results cached by content still hit).
/// @param base File base name.
/// @return The suffix.
std::string module_suffix(const std::string &base) {
  uint32_t h = 2166136261u;
  for (char c : base) {
    h ^= uint8_t(c);
    h *= 16777619u;
  }
  char buf[16];
  snprintf(buf, sizeof buf, "__ym%08x", h);
  return buf;
}

/// Change of brace depth over one line, ignoring "//" comments and quoted text.
/// @param line PTX line.
/// @return Number of '{' minus number of '}'.
int brace_delta(const std::string &line) {
  int d = 0;
  bool quote = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') quote = !quote;
    if (quote) continue;
    if (c == '/' && i + 1 < line.size() && line[i + 1] == '/') break;
    if (c == '{') ++d;
    if (c == '}') --d;
  }
  return d;
}

/// Name of the symbol that a file-scope line declares or defines, or "": a .func, or a
/// .global/.const/.shared variable. *linked tells whether the line gives it linkage (.visible,
/// .extern, .weak or .common); without it the symbol is module-local.
/// @param line PTX line at brace depth 0.
/// @param linked Set to whether the declaration has linkage (.extern counts as linked).
/// @param defines Set to whether the line defines the name for other modules (.visible, .weak,
///        .common).
/// @return The symbol name, or "".
std::string decl_name(const std::string &line, bool *linked, bool *defines) {
  *linked = false;
  *defines = false;
  size_t p = line.find_first_not_of(" \t");
  if (p == std::string::npos || line[p] != '.') return "";
  // An .entry or .func parameter line sits at depth 0 too (".param .u64 .ptr .global .align 1
  // k_param_0,"); its .global is the pointer's state space, not a declaration.
  if (line.compare(p, 6, ".param") == 0 || line.compare(p, 4, ".reg") == 0) return "";
  bool linkage = false, var = false, func = false;
  // Directive words (".global", ".align 8", ".b8", ...) up to the name or the .func keyword.
  while (p < line.size() && line[p] == '.') {
    size_t e = p + 1;
    while (e < line.size() && (ident_char(line[e]) || line[e] == ':')) ++e;
    const std::string w = line.substr(p, e - p);
    p = line.find_first_not_of(" \t", e);
    if (p == std::string::npos) return "";
    if (w == ".visible" || w == ".extern" || w == ".weak" || w == ".common") linkage = true;
    if (w == ".visible" || w == ".weak" || w == ".common") *defines = true;
    if (w == ".global" || w == ".const" || w == ".shared") var = true;
    if (w == ".entry") return "";
    if (w == ".align") {  // its value
      while (p < line.size() && isdigit((unsigned char)line[p])) ++p;
      p = line.find_first_not_of(" \t", p);
      if (p == std::string::npos) return "";
    }
    if (w == ".func") {
      func = true;
      break;
    }
  }
  if (!(var || func)) return "";
  *linked = linkage;
  if (func && p < line.size() && line[p] == '(') {  // return parameter list
    int depth = 0;
    size_t e = p;
    for (; e < line.size(); ++e) {
      if (line[e] == '(') ++depth;
      if (line[e] == ')' && --depth == 0) break;
    }
    if (e >= line.size()) return "";
    p = line.find_first_not_of(" \t", e + 1);
    if (p == std::string::npos) return "";
  }
  if (!ident_start(line[p])) return "";
  size_t e = p;
  while (e < line.size() && ident_char(line[e])) ++e;
  return line.substr(p, e - p);
}

/// Name of the kernel that a file-scope line declares with .entry (any linkage), or "".
/// @param line PTX line at brace depth 0.
/// @return The entry name, or "".
std::string entry_decl_name(const std::string &line) {
  size_t p = line.find_first_not_of(" \t");
  while (p != std::string::npos && p < line.size() && line[p] == '.') {
    size_t e = p + 1;
    while (e < line.size() && ident_char(line[e])) ++e;
    const bool entry = line.compare(p, e - p, ".entry") == 0;
    p = line.find_first_not_of(" \t", e);
    if (p == std::string::npos) return "";
    if (entry) {
      if (!ident_start(line[p])) return "";
      size_t n = p;
      while (n < line.size() && ident_char(line[n])) ++n;
      return line.substr(p, n - p);
    }
  }
  return "";
}

/// Module-local names that a PTX file declares (file-scope lines at brace depth 0).
/// @param path PTX file.
/// @param defined Receives the names the file defines with linkage (.visible, .weak, .common).
/// @param entries Receives the .entry names of the file, in file order.
/// @param hash Receives the 64-bit FNV-1a hash of the file's lines.
/// @return The names (empty when the file cannot be read).
std::set<std::string> local_names(const std::string &path, std::set<std::string> *defined,
                                  std::vector<std::string> *entries, uint64_t *hash) {
  std::set<std::string> names;
  std::ifstream in(path);
  std::string line;
  int depth = 0;
  uint64_t hv = 1469598103934665603ull;
  while (std::getline(in, line)) {
    for (char c : line) {
      hv ^= uint8_t(c);
      hv *= 1099511628211ull;
    }
    hv ^= uint8_t('\n');
    hv *= 1099511628211ull;
    if (depth == 0) {
      bool linked = false, defines = false;
      const std::string n = decl_name(line, &linked, &defines);
      if (!n.empty() && !linked) names.insert(n);
      if (!n.empty() && defines) defined->insert(n);
      const std::string en = entry_decl_name(line);
      if (!en.empty()) entries->push_back(en);
    }
    depth += brace_delta(line);
    if (depth < 0) depth = 0;
  }
  *hash = hv;
  return names;
}

/// Creates directory dir and its parents (like mkdir -p); existing directories are fine.
/// @param dir Directory.
void make_dirs(const std::string &dir) {
  for (size_t i = 1; i <= dir.size(); ++i)
    if (i == dir.size() || dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0755);
}

/// Appends suffix to every identifier of line that is in names. Identifiers after '%'
/// (registers) or '.' (directives and options), quoted text and "//" comments are left alone.
/// @param line PTX line.
/// @param names Module-local names.
/// @param suffix Suffix to append.
/// @return The rewritten line.
std::string rename_line(const std::string &line, const std::set<std::string> &names,
                        const std::string &suffix) {
  std::string out;
  out.reserve(line.size() + 32);
  bool quote = false;
  size_t i = 0;
  while (i < line.size()) {
    const char c = line[i];
    if (quote) {
      out += c;
      if (c == '"') quote = false;
      ++i;
      continue;
    }
    if (c == '"') {
      quote = true;
      out += c;
      ++i;
      continue;
    }
    if (c == '/' && i + 1 < line.size() && line[i + 1] == '/') {
      out.append(line, i, std::string::npos);
      break;
    }
    if (ident_char(c)) {
      size_t e = i;
      while (e < line.size() && ident_char(line[e])) ++e;
      const bool after_sigil = i > 0 && (line[i - 1] == '%' || line[i - 1] == '.');
      out.append(line, i, e - i);
      if (ident_start(c) && !after_sigil && names.count(line.substr(i, e - i))) out += suffix;
      i = e;
      continue;
    }
    out += c;
    ++i;
  }
  return out;
}

}  // namespace

std::string scope_file(const std::string &path, const std::string &unit_in) {
  const char *off = getenv("YASAI_MODULE_SCOPE");
  if (off && atoi(off) == 0) return path;
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto it = r.scoped_path.find(path);
  if (it != r.scoped_path.end()) return it->second;
  uint64_t content_hash = 0;
  std::vector<std::string> entries;
  std::set<std::string> names = local_names(path, &r.defined, &entries, &content_hash);
  // A kernel name that another unit defined first is renamed in this file, so each unit keeps
  // its own kernel; files of one unit (one library's architecture variants) share their names.
  const std::string unit = unit_in.empty() ? file_unit(base_name(path)) : unit_in;
  std::set<std::string> dup_entries;
  for (const std::string &e : entries) {
    auto o = r.entry_unit.find(e);
    if (o == r.entry_unit.end()) r.entry_unit[e] = unit;
    else if (o->second != unit) dup_entries.insert(e);
  }
  names.insert(dup_entries.begin(), dup_entries.end());
  // The renamed kernels depend on which units registered before this file, so they are part of
  // the copy's identity: the shared cache key hashes them after the file's lines.
  for (const std::string &e : dup_entries) {
    for (char c : e) {
      content_hash ^= uint8_t(c);
      content_hash *= 1099511628211ull;
    }
    content_hash ^= uint8_t('\0');
    content_hash *= 1099511628211ull;
  }
  if (names.empty()) {
    r.scoped_path[path] = path;
    return path;
  }
  // The suffix and the copy's name follow the base name, which is the same in every rank
  // directory and run; a second file of the same base name (another directory) takes its full
  // path instead.
  std::string base = base_name(path);
  auto owner = r.base_owner.find(base);
  if (owner != r.base_owner.end() && owner->second != path) {
    std::string flat = path;
    for (char &c : flat)
      if (c == '/') c = '_';
    base = flat;
  }
  r.base_owner[base] = path;
  const std::string suffix = module_suffix(base);
  // The copies go to YASAI_MODULE_SCOPE_DIR when it is set (a cache shared by runs: a copy is
  // named by the hash of its source and its base name, so the same input is written once),
  // otherwise to yasai_module_scope/ in the working directory.
  const char *dir_env = getenv("YASAI_MODULE_SCOPE_DIR");
  const std::string dir = (dir_env && *dir_env) ? std::string(dir_env) : std::string(kScopeDir);
  make_dirs(dir);
  char hx[24];
  snprintf(hx, sizeof hx, "%016llx-", (unsigned long long)content_hash);
  const std::string out_path = dir + "/" + hx + base;
  struct stat st;
  const bool cached = stat(out_path.c_str(), &st) == 0;
  // Written under a unique name (mkstemp) and renamed, so processes that share the directory (the
  // same file, the same content) never read a partial copy. A PID alone is not unique: ranks in
  // two containers can have the same PID in their own PID namespaces.
  std::string tmp;
  if (!cached) {
    std::vector<char> tmpl(out_path.begin(), out_path.end());
    const char suffix_x[] = ".tmpXXXXXX";
    tmpl.insert(tmpl.end(), suffix_x, suffix_x + sizeof suffix_x);  // with the terminating zero
    const int fd = mkstemp(tmpl.data());
    if (fd < 0) {
      fprintf(stderr, "yasai-scope: cannot create a temporary file for %s\n", out_path.c_str());
      abort();
    }
    close(fd);
    tmp = tmpl.data();
    std::ifstream in(path);
    std::ofstream out(tmp);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
      if (!first) out << '\n';
      first = false;
      out << rename_line(line, names, suffix);
    }
    out << '\n';
    if (!out) {
      fprintf(stderr, "yasai-scope: cannot write %s\n", tmp.c_str());
      abort();
    }
  }
  // rename() replaces a copy that another process wrote meanwhile (same name, same content).
  if (!cached && rename(tmp.c_str(), out_path.c_str()) != 0) {
    const bool present = stat(out_path.c_str(), &st) == 0;
    unlink(tmp.c_str());
    if (!present) {
      fprintf(stderr, "yasai-scope: cannot rename %s to %s\n", tmp.c_str(), out_path.c_str());
      abort();
    }
  }
  for (const std::string &n : names) {
    if (dup_entries.count(n)) {
      r.entry_renamed[n].emplace_back(unit, n + suffix);
      r.path_entries[path][n] = n + suffix;
      r.entry_source[n + suffix] = n;
    } else {
      r.renamed[n].emplace_back(base, n + suffix);
    }
  }
  printf("GPGPU-Sim PTX: %zu module-local symbols of %s scoped to the module (%s)\n", names.size(),
         path.c_str(), out_path.c_str());
  if (!dup_entries.empty())
    printf("GPGPU-Sim PTX: %zu kernels of %s are also defined by another module and are renamed "
           "with suffix %s\n", dup_entries.size(), path.c_str(), suffix.c_str());
  r.scoped_path[path] = out_path;
  return out_path;
}

std::string local_name(const std::string &name, const std::string &library) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto it = r.renamed.find(name);
  if (it == r.renamed.end()) return name;
  const std::string lib = base_name(library);
  // A module of the registering library is chosen first. Without one (the library is unknown,
  // or the fat binary was registered under another file name), a single candidate is the
  // registered symbol unless some module defines the name with linkage: another library's local
  // name must not capture a .visible name.
  if (!lib.empty()) {
    const std::string key = library_key(lib);
    for (const auto &cand : it->second)
      if (library_key(cand.first) == key) return cand.second;
  }
  if (it->second.size() == 1 && !r.defined.count(name)) return it->second.front().second;
  return name;
}

std::string module_entry_name(const std::string &path, const std::string &name) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto p = r.path_entries.find(path);
  if (p == r.path_entries.end()) return name;
  auto e = p->second.find(name);
  return e == p->second.end() ? name : e->second;
}

std::string library_entry_name(const std::string &name, const std::string &library) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto it = r.entry_renamed.find(name);
  if (it == r.entry_renamed.end() || library.empty()) return name;
  const std::string key = library_key(base_name(library));
  for (const auto &cand : it->second)
    if (cand.first == key) return cand.second;
  return name;
}

std::string library_unit(const std::string &library) { return library_key(base_name(library)); }

std::string unit_entry_name(const std::string &name, const std::string &unit) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto it = r.entry_renamed.find(name);
  if (it == r.entry_renamed.end() || unit.empty()) return name;
  for (const auto &cand : it->second)
    if (cand.first == unit) return cand.second;
  return name;
}

std::string source_name(const std::string &name) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto it = r.entry_source.find(name);
  return it == r.entry_source.end() ? name : it->second;
}

}  // namespace yasai_scope
