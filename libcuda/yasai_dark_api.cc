// yasai: the driver's private export tables ("dark API") for a statically linked CUDA runtime.
//
// cuGetExportTable(&table, &uuid) returns an array of entry pointers that NVIDIA's CUDA runtime
// calls instead of public driver functions. A shared CUDA runtime goes through the simulator's
// own runtime and never asks; a runtime linked statically into a library does (CUTLASS DSL's
// libcute_dsl_runtime.so asks for three tables while it initializes). The former placeholder
// returned a three-entry table of dummies for every UUID, and the runtime jumped to address 0x1.
//
// The table UUIDs, sizes, entry indices and entry meanings follow ZLUDA's reverse-engineered
// description (https://github.com/vosen/ZLUDA, dark_api/src/lib.rs and zluda/src/impl/driver.rs,
// Apache-2.0 OR MIT). This file reimplements them against the simulator's driver functions; no
// ZLUDA code is copied. As in ZLUDA:
//   * an unknown UUID gets CUDA_ERROR_INVALID_VALUE (the runtime then does without the table);
//   * entry 0 of most tables is the table size in bytes;
//   * entries ZLUDA leaves unimplemented return an error code.
// Entries that no description covers are stubs that print their table and index and abort, so
// a runtime that calls one stops with a message rather than at a junk address.
// Every request is logged to stderr ("yasai export table: ..."), in order.

#include "yasai_dark_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <mutex>
#include <utility>

namespace yasai_dark_api {
namespace {

/// Number of entries of the largest table.
constexpr int kMaxEntries = 26;

/// One export table: its name, UUID (bytes in the order of its textual form) and entries.
struct Table {
  const char *name;           ///< Name used in the log and in ZLUDA's description.
  unsigned char uuid[16];     ///< UUID bytes.
  int size;                   ///< Number of entries.
  const void *entries[kMaxEntries];  ///< Entry pointers (entry 0 may hold the byte size).
};

/// Index of each table in tables() (the stub template needs it at compile time).
enum TableId {
  kCudartInterface, kToolsTls, kToolsRuntimeCallbackHooks, kContextLocalStorage, kCtxCreateBypass,
  kHeapAccess, kDeviceExtendedRt, kIntegrityCheck, kContextChecks, kTableCount
};

/// Names of the tables, by TableId (for the stub messages).
const char *const kTableNames[kTableCount] = {
    "CUDART_INTERFACE", "TOOLS_TLS", "TOOLS_RUNTIME_CALLBACK_HOOKS",
    "CONTEXT_LOCAL_STORAGE_INTERFACE_V0301", "CTX_CREATE_BYPASS", "HEAP_ACCESS",
    "DEVICE_EXTENDED_RT", "INTEGRITY_CHECK", "CONTEXT_CHECKS"};

/// An entry no description covers: prints the table and the index, then aborts.
template <int T, int I>
void undescribed_entry() {
  fprintf(stderr, "yasai export table: %s entry %d is not implemented (called); aborting\n",
          kTableNames[T], I);
  fflush(stderr);
  abort();
}

/// A described entry the simulator does not implement (ZLUDA does not either): logs and returns
/// CUDA_ERROR_NOT_SUPPORTED.
template <int T, int I>
CUresult unsupported_entry() {
  fprintf(stderr, "yasai export table: %s entry %d unsupported, returning CUDA_ERROR_NOT_SUPPORTED\n",
          kTableNames[T], I);
  fflush(stderr);
  return CUDA_ERROR_NOT_SUPPORTED;
}

// ---- CUDART_INTERFACE ----

/// Entry 1: a module from a fat binary wrapper (the runtime's replacement for
/// __cudaRegisterFatBinary). The simulator's module loader takes the wrapper as an image.
/// @param module Receives the module.
/// @param fatbinc_wrapper Fat binary wrapper (magic 0x466243b1).
CUresult get_module_from_cubin(CUmodule *module, const void *fatbinc_wrapper) {
  if (module == NULL || fatbinc_wrapper == NULL) return CUDA_ERROR_INVALID_VALUE;
  return cuModuleLoadData(module, fatbinc_wrapper);
}

/// Entry 2: the primary context of a device (not retained).
/// @param pctx Receives the context.
/// @param dev Device ordinal.
CUresult primary_context(CUcontext *pctx, CUdevice dev) {
  if (pctx == NULL) return CUDA_ERROR_INVALID_VALUE;
  return cuDevicePrimaryCtxRetain(pctx, dev);
}

/// Entry 6: as entry 1, with three further arguments that are null or 0 in the cases described.
CUresult get_module_from_cubin_ext1(CUmodule *module, const void *fatbinc_wrapper, void *arg3,
                                    void *arg4, unsigned arg5) {
  if (arg3 != NULL || arg4 != NULL || arg5 != 0) return CUDA_ERROR_NOT_SUPPORTED;
  return get_module_from_cubin(module, fatbinc_wrapper);
}

/// Entry 7: no observable effect.
void cudart_interface_fn7(size_t) {}

/// Entry 8: a module from a fat binary header (magic 0xba55ed50); further arguments as entry 6.
CUresult get_module_from_cubin_ext2(const void *fatbin_header, CUmodule *module, void *arg3,
                                    void *arg4, unsigned arg5) {
  if (arg3 != NULL || arg4 != NULL || arg5 != 0) return CUDA_ERROR_NOT_SUPPORTED;
  if (module == NULL || fatbin_header == NULL) return CUDA_ERROR_INVALID_VALUE;
  return cuModuleLoadData(module, fatbin_header);
}

/// Entry 12: loads the PTX compiler, libnvvm and the tile IR compiler in the driver; nothing to
/// load here.
CUresult load_compilers() { return CUDA_SUCCESS; }

// ---- TOOLS_RUNTIME_CALLBACK_HOOKS ----

/// Buffers the runtime's tool hooks write into (sizes as described: 1024 and 14 bytes).
unsigned char g_tool_buffer1[1024];
unsigned char g_tool_buffer2[14];

/// Entry 2: the first tool buffer.
void get_tool_buffer1(void **ptr, size_t *size) {
  *ptr = g_tool_buffer1;
  *size = sizeof(g_tool_buffer1);
}

/// Entry 6: the second tool buffer.
void get_tool_buffer2(void **ptr, size_t *size) {
  *ptr = g_tool_buffer2;
  *size = sizeof(g_tool_buffer2);
}

// ---- CONTEXT_LOCAL_STORAGE_INTERFACE_V0301 ----

/// Destructor callback of a stored value (called when the context is reset).
typedef void (*StorageDtor)(CUcontext, void *, void *);

/// Context-local storage: (context, key) -> (value, destructor).
std::map<std::pair<CUcontext, void *>, std::pair<void *, StorageDtor>> &storage() {
  static std::map<std::pair<CUcontext, void *>, std::pair<void *, StorageDtor>> s;
  return s;
}

/// Guards storage().
std::mutex g_storage_mutex;

/// The given context, or the current one when it is null.
/// @param ctx Context or NULL.
/// @return The context to use (NULL when there is none).
CUcontext context_or_current(CUcontext ctx) {
  if (ctx != NULL) return ctx;
  CUcontext cur = NULL;
  if (cuCtxGetCurrent(&cur) != CUDA_SUCCESS || cur == NULL) cuDevicePrimaryCtxRetain(&cur, 0);
  return cur;
}

/// Entry 0: stores value under key in the context.
CUresult storage_put(CUcontext ctx, void *key, void *value, StorageDtor dtor) {
  std::lock_guard<std::mutex> lock(g_storage_mutex);
  storage()[std::make_pair(context_or_current(ctx), key)] = std::make_pair(value, dtor);
  return CUDA_SUCCESS;
}

/// Entry 1: removes key from the context.
CUresult storage_delete(CUcontext ctx, void *key) {
  std::lock_guard<std::mutex> lock(g_storage_mutex);
  storage().erase(std::make_pair(context_or_current(ctx), key));
  return CUDA_SUCCESS;
}

/// Entry 2: the value stored under key in the context (CUDA_ERROR_INVALID_HANDLE when none).
CUresult storage_get(void **value, CUcontext ctx, void *key) {
  std::lock_guard<std::mutex> lock(g_storage_mutex);
  auto it = storage().find(std::make_pair(context_or_current(ctx), key));
  if (it == storage().end()) return CUDA_ERROR_INVALID_HANDLE;
  *value = it->second.first;
  return CUDA_SUCCESS;
}

// ---- CTX_CREATE_BYPASS ----

/// Entry 1: creates a context bypassing the public path; the simulator has one context per
/// process, which is returned.
CUresult ctx_create_bypass(CUcontext *pctx, unsigned flags, CUdevice dev) {
  (void)flags;
  if (pctx == NULL) return CUDA_ERROR_INVALID_VALUE;
  return cuDevicePrimaryCtxRetain(pctx, dev);
}

// ---- CONTEXT_CHECKS ----

/// Entry 2: a context check whose first result is 0 on the devices described.
CUresult context_check(CUcontext, unsigned *result1, const void **) {
  if (result1 != NULL) *result1 = 0;
  return CUDA_SUCCESS;
}

/// Entry 3: returns 0 on the devices described.
unsigned context_check_fn3() { return 0; }

/// Byte size of a table of n entries (entry 0 of the tables that carry it).
const void *size_entry(int n) { return reinterpret_cast<const void *>(sizeof(void *) * n); }

/// Parses the textual UUID form "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" into bytes in that order.
/// @param text UUID text.
/// @param out Receives the 16 bytes.
void parse_uuid(const char *text, unsigned char *out) {
  int n = 0;
  for (const char *p = text; *p && n < 16; ++p) {
    if (*p == '-') continue;
    unsigned v = 0;
    sscanf(p, "%2x", &v);
    out[n++] = (unsigned char)v;
    ++p;
  }
}

/// Fills every entry of a table with the undescribed stub of its index.
template <int T, int... I>
void fill_stubs(Table &t, std::integer_sequence<int, I...>) {
  ((t.entries[I] = reinterpret_cast<const void *>(&undescribed_entry<T, I>)), ...);
}

/// The tables, built once.
Table *tables() {
  static Table t[kTableCount];
  static std::once_flag once;
  std::call_once(once, [] {
    const char *uuids[kTableCount] = {
        "6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9", "42D85A81-23F6-CB47-8298-F6E78A3AECDC",
        "A094798C-2E74-2E74-93F2-0800200C0A66", "C693336E-1121-DF11-A8C3-68F355D89593",
        "0CA50B8C-1004-929A-89A7-D0DF10E77286", "195BCBF4-D67D-024A-ACC5-1D29CEA631AE",
        "B10541E1-F7C7-C74A-9F64-F223BE99F1E2", "D4082055-BDE6-704B-8D34-BA123C66E1F2",
        "263E8860-7CD2-6143-92F6-BBD5006DFA7E"};
    const int sizes[kTableCount] = {13, 4, 7, 4, 2, 3, 26, 3, 4};
    auto seq = std::make_integer_sequence<int, kMaxEntries>{};
    fill_stubs<kCudartInterface>(t[kCudartInterface], seq);
    fill_stubs<kToolsTls>(t[kToolsTls], seq);
    fill_stubs<kToolsRuntimeCallbackHooks>(t[kToolsRuntimeCallbackHooks], seq);
    fill_stubs<kContextLocalStorage>(t[kContextLocalStorage], seq);
    fill_stubs<kCtxCreateBypass>(t[kCtxCreateBypass], seq);
    fill_stubs<kHeapAccess>(t[kHeapAccess], seq);
    fill_stubs<kDeviceExtendedRt>(t[kDeviceExtendedRt], seq);
    fill_stubs<kIntegrityCheck>(t[kIntegrityCheck], seq);
    fill_stubs<kContextChecks>(t[kContextChecks], seq);
    for (int i = 0; i < kTableCount; ++i) {
      t[i].name = kTableNames[i];
      t[i].size = sizes[i];
      parse_uuid(uuids[i], t[i].uuid);
    }
    auto fn = [](auto f) { return reinterpret_cast<const void *>(f); };
    Table &ci = t[kCudartInterface];
    ci.entries[0] = size_entry(13);
    ci.entries[1] = fn(&get_module_from_cubin);
    ci.entries[2] = fn(&primary_context);
    ci.entries[6] = fn(&get_module_from_cubin_ext1);
    ci.entries[7] = fn(&cudart_interface_fn7);
    ci.entries[8] = fn(&get_module_from_cubin_ext2);
    ci.entries[12] = fn(&load_compilers);
    t[kToolsTls].entries[0] = size_entry(4);
    Table &hooks = t[kToolsRuntimeCallbackHooks];
    hooks.entries[0] = size_entry(7);
    hooks.entries[2] = fn(&get_tool_buffer1);
    hooks.entries[6] = fn(&get_tool_buffer2);
    Table &cls = t[kContextLocalStorage];  // this table has no size entry
    cls.entries[0] = fn(&storage_put);
    cls.entries[1] = fn(&storage_delete);
    cls.entries[2] = fn(&storage_get);
    Table &bypass = t[kCtxCreateBypass];
    bypass.entries[0] = size_entry(2);
    bypass.entries[1] = fn(&ctx_create_bypass);
    Table &heap = t[kHeapAccess];
    heap.entries[0] = size_entry(3);
    heap.entries[1] = fn(&unsupported_entry<kHeapAccess, 1>);
    heap.entries[2] = fn(&unsupported_entry<kHeapAccess, 2>);
    Table &ext = t[kDeviceExtendedRt];
    ext.entries[0] = size_entry(26);
    ext.entries[5] = fn(&unsupported_entry<kDeviceExtendedRt, 5>);
    ext.entries[13] = fn(&unsupported_entry<kDeviceExtendedRt, 13>);
    Table &integrity = t[kIntegrityCheck];
    integrity.entries[0] = size_entry(3);
    integrity.entries[1] = fn(&unsupported_entry<kIntegrityCheck, 1>);
    Table &checks = t[kContextChecks];
    checks.entries[0] = size_entry(4);
    checks.entries[2] = fn(&context_check);
    checks.entries[3] = fn(&context_check_fn3);
  });
  return t;
}

/// Number of requests so far (for the log).
int g_requests = 0;

}  // namespace

CUresult get_export_table(const void **table, const CUuuid *id) {
  if (table == NULL || id == NULL) return CUDA_ERROR_INVALID_VALUE;
  const unsigned char *b = reinterpret_cast<const unsigned char *>(id->bytes);
  char text[40];
  snprintf(text, sizeof(text), "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
           b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
           b[14], b[15]);
  Table *t = tables();
  for (int i = 0; i < kTableCount; ++i) {
    if (memcmp(t[i].uuid, b, 16) == 0) {
      fprintf(stderr, "yasai export table: request %d {%s} %s\n", ++g_requests, text, t[i].name);
      fflush(stderr);
      *table = t[i].entries;
      return CUDA_SUCCESS;
    }
  }
  fprintf(stderr, "yasai export table: request %d {%s} unknown, returning CUDA_ERROR_INVALID_VALUE\n",
          ++g_requests, text);
  fflush(stderr);
  *table = NULL;
  return CUDA_ERROR_INVALID_VALUE;
}

}  // namespace yasai_dark_api
