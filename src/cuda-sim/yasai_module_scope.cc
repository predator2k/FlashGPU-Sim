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

/// Module-local names that a PTX file declares (file-scope lines at brace depth 0).
/// @param path PTX file.
/// @param defined Receives the names the file defines with linkage (.visible, .weak, .common).
/// @return The names (empty when the file cannot be read).
std::set<std::string> local_names(const std::string &path, std::set<std::string> *defined) {
  std::set<std::string> names;
  std::ifstream in(path);
  std::string line;
  int depth = 0;
  while (std::getline(in, line)) {
    if (depth == 0) {
      bool linked = false, defines = false;
      const std::string n = decl_name(line, &linked, &defines);
      if (!n.empty() && !linked) names.insert(n);
      if (!n.empty() && defines) defined->insert(n);
    }
    depth += brace_delta(line);
    if (depth < 0) depth = 0;
  }
  return names;
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

std::string scope_file(const std::string &path) {
  const char *off = getenv("YASAI_MODULE_SCOPE");
  if (off && atoi(off) == 0) return path;
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto it = r.scoped_path.find(path);
  if (it != r.scoped_path.end()) return it->second;
  const std::set<std::string> names = local_names(path, &r.defined);
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
  mkdir(kScopeDir, 0755);
  const std::string out_path = std::string(kScopeDir) + "/" + base;
  // Written under a per-process name and renamed, so processes that share a working directory
  // (the same file, the same content) never read a partial copy.
  const std::string tmp = out_path + ".tmp" + std::to_string(getpid());
  {
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
  if (rename(tmp.c_str(), out_path.c_str()) != 0) {
    fprintf(stderr, "yasai-scope: cannot rename %s to %s\n", tmp.c_str(), out_path.c_str());
    abort();
  }
  for (const std::string &n : names) r.renamed[n].emplace_back(base, n + suffix);
  printf("GPGPU-Sim PTX: %zu module-local symbols of %s scoped to the module (%s)\n", names.size(),
         path.c_str(), out_path.c_str());
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

}  // namespace yasai_scope
