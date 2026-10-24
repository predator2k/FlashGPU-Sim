#include "yasai_fakeenv_rec.h"
#include "yasai_jit_image.h"
#include "../src/yasai_subprocess.h"
#include <dlfcn.h>
// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/*
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * George L. Yuan and the University of British Columbia, Vancouver,
 * BC V6T 1Z4, All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and
 * benchmarks/template/ are derived from the CUDA SDK available from
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from
 * src/intersim/ are derived from Booksim (a simulator provided with the
 * textbook "Principles and Practices of Interconnection Networks" available
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by
 * the corresponding legal terms and conditions set forth separately (original
 * copyright notices are left in files from these sources and where we have
 * modified a file our copyright notice appears before the original copyright
 * notice).
 *
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use
 * only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <assert.h>
#include <cctype>
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>  // Apple's version of GLUT is here
#else
#include <GL/gl.h>
#endif
#endif

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "cuda_api.h"
#if defined(__has_include)
#if __has_include("cudaProfiler.h")
#include "cudaProfiler.h"
#else
typedef enum CUoutput_mode_enum {
  CU_OUT_KEY_VALUE_PAIR = 0x00,
  CU_OUT_CSV = 0x01
} CUoutput_mode;
#endif
#else
#include "cudaProfiler.h"
#endif
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
#endif
#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/host_window.h"
#include "yasai_vmm.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpu-sim/flash/tensormap.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/gpgpusim_entrypoint.h"
#include "../src/stream_manager.h"
#include "../src/trace.h"
#include "cuda_api_object.h"
#include "gpgpu_context.h"

typedef cudaKernel_t CUkernel;

#include <pthread.h>
#include <cstdlib>
#include <algorithm>
#include <semaphore.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// SST cycle
extern bool SST_Cycle();

/*DEVICE_BUILTIN*/
struct cudaArray {
  void *devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size;  // in bytes
  unsigned dimensions;
};

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) = v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

cudaError_t g_last_cudaError = cudaSuccess;

void register_ptx_function(const char *name, function_info *impl) {
  // no longer need this
}

#if defined __APPLE__
#define __my_func__ __PRETTY_FUNCTION__
#else
#if defined __cplusplus ? __GNUC_PREREQ(2, 6) : __GNUC_PREREQ(2, 4)
#define __my_func__ __PRETTY_FUNCTION__
#else
#if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#define __my_func__ __func__
#else
#define __my_func__ ((__const char *)0)
#endif
#endif
#endif

struct _cuda_device_id *gpgpu_context::GPGPUSim_Init() {
  _cuda_device_id *the_device = the_gpgpusim->the_cude_device;
  if (!the_device) {
    gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();

    cudaDeviceProp *prop = (cudaDeviceProp *)calloc(sizeof(cudaDeviceProp), 1);
    snprintf(prop->name, sizeof(prop->name), "%s",
             g_gpgpusim_version_string);
    prop->major = the_gpu->compute_capability_major();
    prop->minor = the_gpu->compute_capability_minor();
    prop->totalGlobalMem = 0x80000000 /* 2 GB */;
    prop->memPitch = 0;
    if (prop->major >= 2) {
      prop->maxThreadsPerBlock = 1024;
      prop->maxThreadsDim[0] = 1024;
      prop->maxThreadsDim[1] = 1024;
    } else {
      prop->maxThreadsPerBlock = 512;
      prop->maxThreadsDim[0] = 512;
      prop->maxThreadsDim[1] = 512;
    }

    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 0x40000000;
    prop->maxGridSize[1] = 0x40000000;
    prop->maxGridSize[2] = 0x40000000;
    prop->totalConstMem = 0x40000000;
    prop->textureAlignment = 0;
    //        * TODO: Update the .config and xml files of all GPU config files
    //        with new value of sharedMemPerBlock and regsPerBlock
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
#if (CUDART_VERSION > 5050)
    prop->regsPerMultiprocessor = the_gpu->num_registers_per_core();
    prop->sharedMemPerMultiprocessor = the_gpu->shared_mem_size();
#endif
#if (CUDART_VERSION >= 9000)
    prop->sharedMemPerBlockOptin = the_gpu->shared_mem_per_block_optin();
#endif
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    prop->regsPerBlock = the_gpu->num_registers_per_block();
    prop->warpSize = the_gpu->wrp_size();
#if (CUDART_VERSION < 13000)
    prop->clockRate = the_gpu->shader_clock();
#endif
#if (CUDART_VERSION >= 2010)
    prop->multiProcessorCount = the_gpu->get_config().num_shader();
#endif
#if (CUDART_VERSION >= 4000)
    prop->maxThreadsPerMultiProcessor = the_gpu->threads_per_core();
#endif
    the_gpu->set_prop(prop);
    the_gpgpusim->the_cude_device = new _cuda_device_id(the_gpu);
    the_device = the_gpgpusim->the_cude_device;
  }
  start_sim_thread(1);
  return the_device;
}

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx) {
  // static CUctx_st *the_context = NULL;
  CUctx_st *the_context = ctx->the_gpgpusim->the_context;
  if (the_context == NULL) {
    _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
    ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
    the_context = ctx->the_gpgpusim->the_context;
  }
  return the_context;
}

gpgpu_context *GPGPU_Context();

// yasai: defined further down; the driver cuIpc* functions forward to them.
extern "C" __host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);
extern "C" __host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);
extern "C" __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr);

/// yasai: simulated GPU of the current context (used by yasai_vmm.cc).
gpgpu_t *yasai_current_gpu() {
  return GPGPUSim_Context(GPGPU_Context())->get_device()->get_gpgpu();
}

gpgpu_context *GPGPU_Context() {
  static gpgpu_context *gpgpu_ctx = NULL;
  if (gpgpu_ctx == NULL) {
    gpgpu_ctx = new gpgpu_context();
  }
  return gpgpu_ctx;
}

void ptxinfo_data::ptxinfo_addinfo() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  if (!get_ptxinfo_kname()) {
    /* This info is not per kernel (since CUDA 5.0 some info (e.g. gmem, and
     * cmem) is added at the beginning for the whole binary ) */
    print_ptxinfo();
    context->add_ptxinfo(get_ptxinfo());
    clear_ptxinfo();
    return;
  }
  if (!strcmp("__cuda_dummy_entry__", get_ptxinfo_kname())) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  print_ptxinfo();
  context->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo());
  clear_ptxinfo();
}

void cuda_not_implemented(const char *func, unsigned line) {
  fflush(stdout);
  fflush(stderr);
  printf(
      "\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not "
      "been implemented yet.\n"
      "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n",
      func, __FILE__, line);
  fflush(stdout);
  abort();
}

void announce_call(const char *func) {
  printf("\n\nGPGPU-Sim PTX: CUDA API function \"%s\" has been called.\n",
         func);
  fflush(stdout);
}

#define gpgpusim_ptx_error(msg, ...) \
  gpgpusim_ptx_error_impl(__func__, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define gpgpusim_ptx_assert(cond, msg, ...)                           \
  gpgpusim_ptx_assert_impl((cond), __func__, __FILE__, __LINE__, msg, \
                           ##__VA_ARGS__)

void gpgpusim_ptx_error_impl(const char *func, const char *file, unsigned line,
                             const char *msg, ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  printf("GPGPU-Sim CUDA API: %s\n", buf);
  printf("                    [%s:%u : %s]\n", file, line, func);
  abort();
}

void gpgpusim_ptx_assert_impl(int test_value, const char *func,
                              const char *file, unsigned line, const char *msg,
                              ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  if (test_value == 0) gpgpusim_ptx_error_impl(func, file, line, msg);
}

typedef std::map<unsigned, CUevent_st *> event_tracker_t;

int CUevent_st::m_next_event_uid;
event_tracker_t g_timer_events;

extern int cuobjdump_lex_init(yyscan_t *scanner);
extern void cuobjdump_set_in(FILE *_in_str, yyscan_t yyscanner);
extern int cuobjdump_parse(yyscan_t scanner, struct cuobjdump_parser *parser,
                           std::list<cuobjdumpSection *> &cuobjdumpSectionList);
extern int cuobjdump_lex_destroy(yyscan_t scanner);

enum cuobjdumpSectionType { PTXSECTION = 0, ELFSECTION };

// sectiontype: 0 for ptx, 1 for elf
void addCuobjdumpSection(int sectiontype,
                         std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (sectiontype)
    cuobjdumpSectionList.push_front(new cuobjdumpELFSection());
  else
    cuobjdumpSectionList.push_front(new cuobjdumpPTXSection());
  printf("## Adding new section %s\n", sectiontype ? "ELF" : "PTX");
}

void setCuobjdumparch(const char *arch,
                      std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  unsigned archnum;
  sscanf(arch, "sm_%u", &archnum);
  assert(archnum && "cannot have sm_0");
  printf("Adding arch: %s\n", arch);
  cuobjdumpSectionList.front()->setArch(archnum);
}

void setCuobjdumpidentifier(
    const char *identifier,
    std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  printf("Adding identifier: %s\n", identifier);
  cuobjdumpSectionList.front()->setIdentifier(identifier);
}

void setCuobjdumpptxfilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  printf("Adding ptx filename: %s\n", filename);
  cuobjdumpSection *x = cuobjdumpSectionList.front();
  if (dynamic_cast<cuobjdumpPTXSection *>(x) == NULL) {
    assert(0 &&
           "You shouldn't be trying to add a ptxfilename to an elf section");
  }
  (dynamic_cast<cuobjdumpPTXSection *>(x))->setPTXfilename(filename);
}

void setCuobjdumpelffilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()) ==
      NULL) {
    assert(0 &&
           "You shouldn't be trying to add a elffilename to an ptx section");
  }
  (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()))
      ->setELFfilename(filename);
}

void setCuobjdumpsassfilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()) ==
      NULL) {
    assert(0 &&
           "You shouldn't be trying to add a sassfilename to an ptx section");
  }
  (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()))
      ->setSASSfilename(filename);
}

//! Return the executable file of the process containing the PTX/SASS code
//!
//! This Function returns the executable file ran by the process.  This
//! executable is supposed to contain the PTX/SASS code.  It provides workaround
//! for processes running on valgrind by dereferencing /proc/<pid>/exe within
//! the GPGPU-Sim process before calling cuobjdump to extract PTX/SASS.  This is
//! needed because valgrind uses x86 emulation to detect memory leak.  Other
//! processes (e.g. cuobjdump) reading /proc/<pid>/exe will see the emulator
//! executable instead of the application binary.
//!
// In SST need the string to pass the binary information
// as we cannot get it from /proc/self/exe
std::string get_app_binary(const char *fn) {
  printf("self exe links to: %s\n", fn);
  return fn;
}

std::string get_app_binary() {
  char self_exe_path[1025];
#ifdef __APPLE__
  uint32_t size = sizeof(self_exe_path);
  if (_NSGetExecutablePath(self_exe_path, &size) != 0) {
    printf("GPGPU-Sim ** ERROR: _NSGetExecutablePath input buffer too small\n");
    exit(1);
  }
#else
  std::stringstream exec_link;
  exec_link << "/proc/self/exe";

  // ssize_t path_length = readlink(exec_link.str().c_str(), self_exe_path, 1024);
  ssize_t path_length = readlink("/proc/self/exe", self_exe_path, 1024);
  assert(path_length != -1);
  self_exe_path[path_length] = '\0';
#endif

  printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}

// above func gives abs path whereas this give just the name of application.
char *get_app_binary_name(std::string abs_path) {
  char *self_exe_path = NULL;
#ifdef __APPLE__
  // TODO: get apple device and check the result.
  printf("WARNING: not tested for Apple-mac devices \n");
  abort();
#else
  char *buf = strdup(abs_path.c_str());
  char *token = strtok(buf, "/");
  while (token != NULL) {
    self_exe_path = token;
    token = strtok(NULL, "/");
  }
#endif
  self_exe_path = strtok(self_exe_path, ".");
  printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}

// yasai: cluster shape of the next launch, set from cudaLaunchKernelExC / cuLaunchKernelEx
// attributes and consumed (reset to 1,1,1) when the launch creates its kernel_info_t.
static dim3 g_yasai_pending_cluster(1, 1, 1);

static int get_app_cuda_version_internal(std::string app_binary) {
  int app_cuda_version = 0;
  // yasai: an interpreter (python running PyTorch) does not link libcudart, so the version is
  // read from YASAI_APP_CUDA_VERSION if set, else from the library named by PYTORCH_BIN.
  if (const char *v = getenv("YASAI_APP_CUDA_VERSION")) {
    if (atoi(v) > 0) return atoi(v);
  }
  if (const char *pb = getenv("PYTORCH_BIN")) {
    // yasai: PYTORCH_BIN may list several libraries separated by ':'; the first one is probed.
    if (strlen(pb) != 0) app_binary = std::string(pb).substr(0, std::string(pb).find(':'));
  }
  // yasai: one probe per binary; the probe runs ldd, readelf and strings over the binary, which
  // takes seconds on libtorch_cuda.so, and a PyTorch process registers dozens of fat binaries.
  static std::map<std::string, int> cached_versions;
  auto cached = cached_versions.find(app_binary);
  if (cached != cached_versions.end()) return cached->second;
  struct yasai_version_cache_store {
    std::map<std::string, int> &m;
    const std::string &key;
    int &value;
    ~yasai_version_cache_store() { if (value) m[key] = value; }
  } yasai_store{cached_versions, app_binary, app_cuda_version};
  char fname[1024];
  snprintf(fname, 1024, "_app_cuda_version_XXXXXX");
  int fd = mkstemp(fname);
  close(fd);
  // Weili: Add way to extract CUDA version information from Balar Vanadis
  // binary (stored as a const string)
  // Try ldd first, then fall back to readelf for cases where ldd doesn't
  // display libcudart (e.g., gpgpu-sim's libcudart.so has no versioned SONAME)
  std::string app_cuda_version_command =
      "{ ldd " + app_binary +
      " | grep libcudart.so | sed  's/.*libcudart.so.\\(.*\\) =>.*/\\1/'; "
      "readelf -d " +
      app_binary +
      " 2>/dev/null | grep 'libcudart.so\\.' | sed "
      "'s/.*libcudart.so\\.\\([0-9][0-9.]*\\)\\].*/\\1/'; "
      "strings " +
      app_binary +
      " | grep libcudart_vanadis.a | sed  "
      "'s/.*libcudart_vanadis.a.\\(.*\\)/\\1/'; } | head -1 > " +
      fname;
  int res = yasai_system(app_cuda_version_command.c_str());
  if (res == -1) {
    printf("Error - Cannot detect the app's CUDA version. Command: %s\n",
           app_cuda_version_command.c_str());
    exit(1);
  }
  FILE *cmd = fopen(fname, "r");
  char buf[256];
  while (fgets(buf, sizeof(buf), cmd) != 0) {
    std::cout << buf;
    app_cuda_version = atoi(buf);
  }
  fclose(cmd);
  if (app_cuda_version == 0) {
    printf("Error - Cannot detect the app's CUDA version. Command: %s\n",
           app_cuda_version_command.c_str());
    exit(1);
  }
  return app_cuda_version;
}

static int get_app_cuda_version(const char *fn) {
  // Use for other simulator integration
  std::string app_binary = get_app_binary(fn);
  return get_app_cuda_version_internal(app_binary);
}

static int get_app_cuda_version() {
  std::string app_binary = get_app_binary();
  return get_app_cuda_version_internal(app_binary);
}

//! Keep track of the association between filename and cubin handle
void cuda_runtime_api::cuobjdumpRegisterFatBinary(unsigned int handle,
                                                  const char *filename,
                                                  CUctx_st *context) {
  fatbinmap[handle] = filename;
}

/*******************************************************************************
 * Add internal cuda runtime API call to accept gpgpu_context *
 *******************************************************************************/
cudaError_t cudaSetDeviceInternal(int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // set the active device to run cuda
  if (device <= ctx->GPGPUSim_Init()->num_devices()) {
    ctx->api->g_active_device = device;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

cudaError_t cudaGetDeviceInternal(int *device,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *device = ctx->api->g_active_device;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimitInternal(
    size_t *pValue, cudaLimit limit, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  const struct cudaDeviceProp *prop = dev->get_prop();
  const gpgpu_sim_config &config = dev->get_gpgpu()->get_config();
  switch (limit) {
    case 0:  // cudaLimitStackSize
      *pValue = config.stack_limit();
      break;
    case 2:  // cudaLimitMallocHeapSize
      *pValue = config.heap_limit();
      break;
#if (CUDART_VERSION > 5050)
    case 3:  // cudaLimitDevRuntimeSyncDepth
      if (prop->major > 2) {
        *pValue = config.sync_depth_limit();
        break;
      } else {
        printf("ERROR:Limit %d is not supported on this architecture \n",
               limit);
        abort();
      }
    case 4:  // cudaLimitDevRuntimePendingLaunchCount
      if (prop->major > 2) {
        *pValue = config.pending_launch_count_limit();
        break;
      } else {
        printf("ERROR:Limit %d is not supported on this architecture \n",
               limit);
        abort();
      }
#endif
    default:
      printf("ERROR:Limit %d unimplemented \n", limit);
      abort();
  }
  return g_last_cudaError = cudaSuccess;
}

// Helper function to allocate unique fatbin/module handles
static unsigned get_next_fat_bin_handle() {
  static unsigned next_fat_bin_handle = 1;
  unsigned handle = next_fat_bin_handle;
  next_fat_bin_handle++;
  return handle;
}

static bool file_exists(const std::string &path) {
  std::ifstream file(path.c_str());
  return file.good();
}

static std::string sidecar_ptx_for_module(const char *module_path) {
  std::string path(module_path);
  const std::string cubin_suffix = ".cubin";
  if (path.length() <= cubin_suffix.length()) return "";
  if (path.compare(path.length() - cubin_suffix.length(), cubin_suffix.length(),
                   cubin_suffix) != 0) {
    return "";
  }
  std::string ptx_path =
      path.substr(0, path.length() - cubin_suffix.length()) + ".ptx";
  return file_exists(ptx_path) ? ptx_path : "";
}

static std::string ptx_target_from_file(const std::string &ptx_path) {
  std::ifstream file(ptx_path.c_str());
  std::string line;
  while (std::getline(file, line)) {
    size_t target_pos = line.find(".target");
    if (target_pos == std::string::npos) continue;
    size_t sm_pos = line.find("sm_", target_pos);
    if (sm_pos == std::string::npos) continue;
    size_t end = line.find_first_of(", \t\r\n", sm_pos);
    return line.substr(sm_pos, end == std::string::npos ? std::string::npos
                                                        : end - sm_pos);
  }
  return "";
}

static std::string get_cuda_library_ptx_path() {
  const char *ptx_path = getenv("GPGPUSIM_CUDA_LIBRARY_PTX");
  if (ptx_path != NULL && strlen(ptx_path) != 0) return std::string(ptx_path);

  ptx_path = getenv("PTX_SIM_KERNELFILE");
  if (ptx_path != NULL && strlen(ptx_path) != 0) return std::string(ptx_path);

  return std::string();
}

static CUresult load_module_from_ptx_file(CUmodule *module,
                                          const std::string &ptx_path,
                                          const char *caller) {
  gpgpu_context *ctx = GPGPU_Context();
  if (!ctx) {
    printf("ERROR: No active GPGPU context\n");
    return CUDA_ERROR_INVALID_HANDLE;
  }

  if (ptx_path.empty() || !file_exists(ptx_path)) {
    printf("GPGPU-Sim PTX: ERROR: %s sidecar PTX not found: %s\n", caller,
           ptx_path.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUctx_st *context = GPGPUSim_Context(ctx);
  unsigned module_handle = get_next_fat_bin_handle();
  printf("GPGPU-Sim PTX: %s loading module handle %u using PTX sidecar %s\n",
         caller, module_handle, ptx_path.c_str());

  symbol_table *symtab =
      ctx->gpgpu_ptx_sim_load_ptx_from_filename(ptx_path.c_str());
  context->add_binary(symtab, module_handle);
  ctx->api->fatbinmap[module_handle] = ptx_path;
  ctx->api->fatbin_registered[module_handle] = true;
  ctx->api->name_symtab[ptx_path] = symtab;

  std::string arch = ptx_target_from_file(ptx_path);
  if (arch.empty()) {
    unsigned forced_capability =
        context->get_device()->get_gpgpu()->get_config().get_forced_max_capability();
    if (forced_capability == 0) forced_capability = 20;
    arch = "sm_" + std::to_string(forced_capability);
  }
  ctx->gpgpu_ptx_info_load_from_filename(ptx_path.c_str(), arch.c_str());

  ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT,
                                context->get_device()->get_gpgpu());
  ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                           context->get_device()->get_gpgpu());

  *module = (CUmodule)(unsigned long long)module_handle;
  return CUDA_SUCCESS;
}

static CUresult load_module_from_cuda_library_ptx(CUmodule *module,
                                                  const char *caller);

// yasai: cuModuleLoadData and cuModuleLoadDataEx load the image they are given (Triton,
// DeepGEMM and CuTe DSL modules): a PTX image directly, a fat binary through its embedded PTX,
// an ELF cubin through the PTX that the JIT caches keep (yasai_jit_image.h). The PTX sidecar of
// GPGPUSIM_CUDA_LIBRARY_PTX / PTX_SIM_KERNELFILE, when its file exists, still takes precedence; an image
// without PTX falls back to it as before.
//   module  receives the module handle
//   image   the image the caller passed
//   caller  name of the API function (for the log)
static CUresult yasai_load_module_image(CUmodule *module, const void *image, const char *caller) {
  // setup_environment always exports PTX_SIM_KERNELFILE=_1.ptx, so the sidecar counts only
  // when its file exists.
  const std::string sidecar = get_cuda_library_ptx_path();
  if (!sidecar.empty() && file_exists(sidecar))
    return load_module_from_cuda_library_ptx(module, caller);
  gpgpu_context *ctx = GPGPU_Context();
  if (!ctx) return CUDA_ERROR_INVALID_HANDLE;
  const unsigned cap = GPGPUSim_Context(ctx)->get_device()->get_gpgpu()->get_config().get_forced_max_capability();
  std::string how;
  const std::string ptx = yasai_jit::ptx_for_image(image, cap, &how);
  printf("GPGPU-Sim PTX: %s: %s\n", caller, how.c_str());
  if (ptx.empty()) return CUDA_ERROR_NO_BINARY_FOR_GPU;
  return load_module_from_ptx_file(module, ptx, caller);
}

static CUresult load_module_from_cuda_library_ptx(CUmodule *module,
                                                  const char *caller) {
  const std::string ptx_path = get_cuda_library_ptx_path();
  if (ptx_path.empty()) {
    printf(
        "GPGPU-Sim PTX: ERROR: %s needs GPGPUSIM_CUDA_LIBRARY_PTX or "
        "PTX_SIM_KERNELFILE to point at the PTX sidecar for an in-memory CUDA "
        "library\n",
        caller);
    return CUDA_ERROR_INVALID_VALUE;
  }
  return load_module_from_ptx_file(module, ptx_path, caller);
}

// yasai: indexes the PTX of a library registered after start-up (defined after the lazy
// loading code).
extern "C" {
static void yasai_register_library_ptx(gpgpu_context *ctx, unsigned handle, void *fatCubin);
}

// Internal implementation for cudaRegisterFatBiaryInternal
void **cudaRegisterFatBiaryInternal_impl(
    void *fatCubin, gpgpu_context *gpgpu_ctx, std::string &app_binary_path,
    int app_cuda_version,
    std::function<void(gpgpu_context *)> ctx_cuobjdumpInit_func) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION < 2010)
  printf(
      "GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or "
      "higher\n");
  exit(1);
#endif
  CUctx_st *context = GPGPUSim_Context(ctx);
  if (context->get_device()->get_gpgpu()->get_config().use_cuobjdump()) {
    // The following workaround has only been verified on 64-bit systems.
    if (sizeof(void *) == 4)
      printf(
          "GPGPU-Sim PTX: FatBin file name extraction has not been tested on "
          "32-bit system.\n");

    // This code will get the CUDA version the app was compiled with.
    // We need this to determine how to handle the parsing of the binary.
    // Making this a runtime variable based on the app, enables GPGPU-Sim
    // compiled with a newer version of CUDA to run apps compiled with older
    // versions of CUDA. This is especially useful for PTXPLUS execution.
    // Skip cuda version check for pytorch application
    size_t pos = app_binary_path.find("python");
    if (pos == std::string::npos) {
      // Not pytorch app : checking cuda version
      if (app_cuda_version != CUDART_VERSION / 1000) {
        printf(
            "GPGPU-Sim PTX: warning -- app CUDA major version %d differs from "
            "simulator CUDA major version %d; continuing with compatibility "
            "shim\n",
            app_cuda_version, CUDART_VERSION / 1000);
      }
    }

    // int app_cuda_version = get_app_cuda_version();
    // assert( app_cuda_version == CUDART_VERSION / 1000  && "The app must be
    // compiled with same major version as the simulator." );
    const char *filename;
#if CUDART_VERSION < 6000
    // FatBin handle from the .fatbin.c file (one of the intermediate files
    // generated by NVCC)
    typedef struct {
      int m;
      int v;
      const unsigned long long *d;
      char *f;
    } __fatDeviceText __attribute__((aligned(8)));
    __fatDeviceText *fatDeviceText = (__fatDeviceText *)fatCubin;

    // Extract the source code file name that generate the given FatBin.
    // - Obtains the pointer to the actual fatbin structure from the FatBin
    // handle (fatCubin).
    // - An integer inside the fatbin structure contains the relative offset to
    // the source code file name.
    // - This offset differs among different CUDA and GCC versions.
    char *pfatbin = (char *)fatDeviceText->d;
    int offset = *((int *)(pfatbin + 48));
    filename = (pfatbin + 16 + offset);
#else
    filename = "default";
#endif

    // The extracted file name is associated with a fat_cubin_handle passed
    // into cudaLaunch().  Inside cudaLaunch(), the associated file name is
    // used to find the PTX/SASS section from cuobjdump, which contains the
    // PTX/SASS code for the launched kernel function.
    // This allows us to work around the fact that cuobjdump only outputs the
    // file name associated with each section.
    unsigned long long fat_cubin_handle = get_next_fat_bin_handle();
    printf(
        "GPGPU-Sim PTX: __cudaRegisterFatBinary, fat_cubin_handle = %llu, "
        "filename=%s\n",
        fat_cubin_handle, filename);
    /*!
     * This function extracts all data from all files in first call
     * then for next calls, only returns the appropriate number
     */
    assert(fat_cubin_handle >= 1);
    if (fat_cubin_handle == 1) ctx_cuobjdumpInit_func(ctx);
    ctx->api->cuobjdumpRegisterFatBinary(fat_cubin_handle, filename, context);
    yasai_register_library_ptx(ctx, fat_cubin_handle, fatCubin);

    return (void **)fat_cubin_handle;
  }
#if (CUDART_VERSION < 8000)
  else {
    static unsigned source_num = 1;
    unsigned long long fat_cubin_handle = next_fat_bin_handle++;
    __cudaFatCudaBinary *info = (__cudaFatCudaBinary *)fatCubin;
    assert(info->version >= 3);
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 0;
    unsigned selected_capability = 0;
    bool found = false;
    unsigned forced_max_capability = context->get_device()
                                         ->get_gpgpu()
                                         ->get_config()
                                         .get_forced_max_capability();
    if (!info->ptx) {
      printf(
          "ERROR: Cannot find ptx code in cubin file\n"
          "\tIf you are using CUDA 4.0 or higher, please enable "
          "-gpgpu_ptx_use_cuobjdump or downgrade to CUDA 3.1\n");
      exit(1);
    }
    while (info->ptx[num_ptx_versions].gpuProfileName != NULL) {
      unsigned capability = 0;
      sscanf(info->ptx[num_ptx_versions].gpuProfileName, "compute_%u",
             &capability);
      printf(
          "GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for "
          "'%s', ",
          info->ident);
      printf("capability = %s\n", info->ptx[num_ptx_versions].gpuProfileName);
      if (forced_max_capability) {
        if (capability > max_capability &&
            capability <= forced_max_capability) {
          found = true;
          max_capability = capability;
          selected_capability = num_ptx_versions;
        }
      } else {
        if (capability > max_capability) {
          found = true;
          max_capability = capability;
          selected_capability = num_ptx_versions;
        }
      }
      num_ptx_versions++;
    }
    if (found) {
      printf("GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
             info->ident, info->ptx[selected_capability].gpuProfileName);
      symbol_table *symtab;
      const char *ptx = info->ptx[selected_capability].ptx;
      if (context->get_device()
              ->get_gpgpu()
              ->get_config()
              .convert_to_ptxplus()) {
        printf(
            "GPGPU-Sim PTX: ERROR ** PTXPlus is only supported through "
            "cuobjdump\n"
            "\tEither enable cuobjdump or disable PTXPlus in your "
            "configuration file\n");
        exit(1);
      } else {
        symtab = ctx->gpgpu_ptx_sim_load_ptx_from_string(ptx, source_num);
        context->add_binary(symtab, fat_cubin_handle);
        ctx->gpgpu_ptxinfo_load_from_string(ptx, source_num, max_capability,
                                            context->no_of_ptx);
      }
      source_num++;
      ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT,
                                    context->get_device()->get_gpgpu());
      ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                               context->get_device()->get_gpgpu());
    } else {
      printf(
          "GPGPU-Sim PTX: warning -- did not find an appropriate PTX in "
          "cubin\n");
    }
    return (void **)fat_cubin_handle;
  }
#else
  else {
    printf("ERROR **  __cudaRegisterFatBinary() needs to be updated\n");
    abort();
  }
#endif
}

void **cudaRegisterFatBinaryInternal(const char *fn, void *fatCubin,
                                     gpgpu_context *gpgpu_ctx) {
  std::string app_binary_path = get_app_binary(fn);
  int app_cuda_version = get_app_cuda_version(fn);
  auto ctx_cuobjdumpInit = [=](gpgpu_context *ctx) {
    ctx->api->cuobjdumpInit(fn);
  };
  return cudaRegisterFatBiaryInternal_impl(fatCubin, gpgpu_ctx, app_binary_path,
                                           app_cuda_version, ctx_cuobjdumpInit);
}

void **cudaRegisterFatBinaryInternal(void *fatCubin,
                                     gpgpu_context *gpgpu_ctx = NULL) {
  std::string app_binary_path = get_app_binary();
  int app_cuda_version = get_app_cuda_version();
  auto ctx_cuobjdumpInit = [](gpgpu_context *ctx) {
    ctx->api->cuobjdumpInit();
  };
  return cudaRegisterFatBiaryInternal_impl(fatCubin, gpgpu_ctx, app_binary_path,
                                           app_cuda_version, ctx_cuobjdumpInit);
}

void cudaRegisterFunctionInternal(void **fatCubinHandle, const char *hostFun,
                                  char *deviceFun, const char *deviceName,
                                  int thread_limit, uint3 *tid, uint3 *bid,
                                  dim3 *bDim, dim3 *gDim,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;
  printf(
      "GPGPU-Sim PTX: __cudaRegisterFunction %s : hostFun 0x%p, "
      "fat_cubin_handle = %u\n",
      deviceFun, hostFun, fat_cubin_handle);
  if (context->get_device()->get_gpgpu()->get_config().use_cuobjdump())
    ctx->cuobjdumpParseBinary(fat_cubin_handle);
  context->register_function(fat_cubin_handle, hostFun, deviceFun);
}

void cudaRegisterVarInternal(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (GPGPUSim_Context(ctx)
          ->get_device()
          ->get_gpgpu()
          ->get_config()
          .use_cuobjdump())
    ctx->cuobjdumpParseBinary((unsigned)(unsigned long long)fatCubinHandle);
  fflush(stdout);
  if (constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_const_variable(hostVar, deviceName,
                                                         size);
  } else if (!constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_global_variable(hostVar, deviceName,
                                                          size);
  } else
    cuda_not_implemented(__my_func__, __LINE__);
}

cudaError_t cudaConfigureCallInternal(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
    printf(
        "GPGPU-Sim PTX: cudaConfigureCall gridDim=(%u,%u,%u) "
        "blockDim=(%u,%u,%u) sharedMem=%zu stream=0x%p\n",
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
        sharedMem, stream);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  ctx->api->g_cuda_launch_stack.push_back(
      kernel_config(gridDim, blockDim, sharedMem, s));
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceCountInternal(int *count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *count = dev->num_devices();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDevicePropertiesInternal(
    struct cudaDeviceProp *prop, int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  if (device <= dev->num_devices()) {
    *prop = *dev->get_prop();
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

__host__ cudaError_t CUDARTAPI
cudaChooseDeviceInternal(int *device, const struct cudaDeviceProp *prop,
                         gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *device = dev->get_id();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaSetupArgumentInternal(const void *arg, size_t size,
                                      size_t offset,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config &config = ctx->api->g_cuda_launch_stack.back();
  config.set_arg(arg, size, offset);
  printf(
      "GPGPU-Sim PTX: Setting up arguments for %zu bytes starting at "
      "0x%llx..\n",
      size, (unsigned long long)arg);

  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaLaunchInternalImpl(const char *hostFun, gpgpu_context *gpgpu_ctx);

// yasai: every kernel launch (cudaLaunchKernel, cudaLaunchKernelExC, cuLaunchKernel,
// cuLaunchKernelEx) reaches this function; with YASAI_FAKEENV_OUT set it records an api and a
// kernel record (yasai_fakeenv_rec.h) around the launch.
cudaError_t cudaLaunchInternal(const char *hostFun, gpgpu_context *gpgpu_ctx = NULL) {
  yasai_rec::Recorder &rec = yasai_rec::Recorder::get();
  gpgpu_context *ctx = gpgpu_ctx ? gpgpu_ctx : GPGPU_Context();
  if (!rec.on() || ctx->api->g_cuda_launch_stack.empty())
    return cudaLaunchInternalImpl(hostFun, gpgpu_ctx);
  kernel_config config = ctx->api->g_cuda_launch_stack.back();
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = rec.corr();
  const cudaError_t ret = cudaLaunchInternalImpl(hostFun, gpgpu_ctx);
  const uint64_t t1 = yasai_rec::mono_ns();
  function_info *entry = GPGPUSim_Context(ctx)->get_kernel(hostFun);
  const std::string name = entry ? entry->get_name() : std::string("unknown");
  const gpgpu_ptx_sim_info *info = entry ? entry->get_kernel_info() : NULL;
  const dim3 g = config.grid_dim(), b = config.block_dim();
  rec.api("cudaLaunchKernel", t0, t1, corr, int(ret));
  rec.write("\"rec\":\"kernel\",\"start\":%llu,\"end\":%llu,\"dev\":0,\"ctx\":1,\"stream\":%u,"
            "\"corr\":%u,\"name\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
            "\"cluster\":[1,1,1],\"smem_static\":%u,\"smem_dynamic\":%zu,\"regs\":%u,"
            "\"local_per_thread\":0,\"graph_id\":0,\"graph_node\":0,\"launch_type\":0,"
            "\"queued\":0,\"submitted\":0,\"device_launched\":0",
            (unsigned long long)t0, (unsigned long long)t1,
            rec.stream_id(config.get_stream()), corr, yasai_rec::esc(name.c_str()).c_str(),
            g.x, g.y, g.z, b.x, b.y, b.z, info ? info->smem : 0u, (size_t)config.shared_mem(),
            info ? info->regs : 0u);
  return ret;
}

cudaError_t cudaLaunchInternalImpl(const char *hostFun, gpgpu_context *gpgpu_ctx) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &(ctx->func_sim->g_ptx_sim_mode));
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config config = ctx->api->g_cuda_launch_stack.back();
  {
    dim3 gridDim = config.grid_dim();
    dim3 blockDim = config.block_dim();
    if (gridDim.x * gridDim.y * gridDim.z == 0 ||
        blockDim.x * blockDim.y * blockDim.z == 0) {
      // can't launch
      printf("can't launch a empty kernel\n");
      ctx->api->g_cuda_launch_stack.pop_back();
      return g_last_cudaError = cudaErrorInvalidConfiguration;
    }
  }
  struct CUstream_st *stream = config.get_stream();

  function_info *entry = context->get_kernel(hostFun);
  const struct gpgpu_ptx_sim_info *kernel_info = entry->get_kernel_info();
  unsigned static_smem = kernel_info ? kernel_info->smem : 0;
  size_t dynamic_smem = config.shared_mem();
  gpgpu_sim *device_gpu = context->get_device()->get_gpgpu();
  bool opted_in =
      device_gpu->has_kernel_max_dynamic_smem(entry->get_name());
  unsigned dynamic_limit = 0;
  unsigned total_limit = 0;
  if (opted_in) {
    dynamic_limit =
        device_gpu->get_kernel_max_dynamic_smem(entry->get_name());
    total_limit = device_gpu->shared_mem_per_block_optin();
  } else {
    total_limit = device_gpu->shared_mem_per_block();
    dynamic_limit =
        static_smem < total_limit ? total_limit - static_smem : 0;
  }
  unsigned long long total_smem =
      static_cast<unsigned long long>(static_smem) + dynamic_smem;
  if (dynamic_smem > dynamic_limit || total_smem > total_limit) {
    printf("GPGPU-Sim PTX: kernel launch dynamic shared memory invalid for "
           "'%s': static=%u, dynamic=%zu, max_dynamic=%u, max_total=%u%s\n",
           entry->get_name().c_str(), static_smem, dynamic_smem, dynamic_limit,
           total_limit, opted_in ? " (opt-in)" : " (default)");
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaErrorInvalidConfiguration;
  }

  printf("\nGPGPU-Sim PTX: cudaLaunch for 0x%p (mode=%s) on stream %u\n",
         hostFun,
         (ctx->func_sim->g_ptx_sim_mode) ? "functional simulation"
                                         : "performance simulation",
         stream ? stream->get_uid() : 0);
  kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      hostFun, config.get_args(), config.grid_dim(), config.block_dim(),
      context);
  // yasai: cluster shape from the launch attribute, else from .reqnctapercluster.
  {
    dim3 cd = g_yasai_pending_cluster;
    g_yasai_pending_cluster = dim3(1, 1, 1);
    const function_info *e = grid->entry();
    if (cd.x * cd.y * cd.z <= 1 && e->req_cluster_dim(0) > 0)
      cd = dim3(e->req_cluster_dim(0), e->req_cluster_dim(1) ? e->req_cluster_dim(1) : 1,
                e->req_cluster_dim(2) ? e->req_cluster_dim(2) : 1);
    if (cd.x == 0) cd.x = 1;
    if (cd.y == 0) cd.y = 1;
    if (cd.z == 0) cd.z = 1;
    grid->set_cluster_dim(cd);
    if (cd.x * cd.y * cd.z > 1)
      printf("GPGPU-Sim PTX: cluster shape (%u,%u,%u) for '%s'\n", cd.x, cd.y, cd.z,
             grid->name().c_str());
  }

  // Handle dynamic shared memory for extern shared symbols and record the
  // per-launch dynamic shared memory size on the kernel grid object
  size_t shared_mem_size = config.shared_mem();
  grid->set_dynamic_smem((unsigned)shared_mem_size);
  if (shared_mem_size > 0) {
    function_info *func_info = grid->entry();
    func_info->alloc_dyn_shared_mem(shared_mem_size);
  }

  // do dynamic PDOM analysis for performance simulation scenario
  std::string kname = grid->name();
  function_info *kernel_func_info = grid->entry();
  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kname.c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kname.c_str());
    kernel_func_info->do_pdom();
    kernel_func_info->set_pdom();
  }
  dim3 gridDim = config.grid_dim();
  dim3 blockDim = config.block_dim();

  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();
  class memory_space *global_mem;
  global_mem = gpu->get_global_memory();

  if (gpu->resume_option == 1 && (grid->get_uid() == gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    for (unsigned i = 0; i < gpu->resume_CTA; i++) grid->increment_cta_id();
  }
  if (gpu->resume_option == 1 && (grid->get_uid() < gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    printf("Skipping kernel %d as resuming from kernel %d\n", grid->get_uid(),
           gpu->resume_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  if (gpu->checkpoint_option == 1 &&
      (grid->get_uid() > (unsigned)gpu->checkpoint_kernel)) {
    printf("Skipping kernel %d as checkpoint from kernel %d\n", grid->get_uid(),
           gpu->checkpoint_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  printf(
      "GPGPU-Sim PTX: pushing kernel \'%s\' to stream %u, gridDim= (%u,%u,%u) "
      "blockDim = (%u,%u,%u) \n",
      kname.c_str(), stream ? stream->get_uid() : 0, gridDim.x, gridDim.y,
      gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  printf("GPGPU-Sim PTX: runtime launch kernel mangled='%s'\n",
         kname.c_str());
  stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
  ctx->the_gpgpusim->g_stream_manager->push(op);
  ctx->api->g_cuda_launch_stack.pop_back();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaMallocInternal(void **devPtr, size_t size,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  // yasai: memfd-backed host memory at the device address (yasai_vmm::malloc_device), unless
  // YASAI_DEVMEM_HOST=0.
  *devPtr = reinterpret_cast<void *>(
      yasai_vmm::malloc_device(context->get_device()->get_gpgpu(), size));
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*devPtr);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*devPtr] = size;
  }
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

cudaError_t cudaMallocHostInternal(void **ptr, size_t size,
                                   gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *ptr = malloc(size);
  if (*ptr) {
    // track pinned memory size allocated in the host so that same amount of
    // memory is also allocated in GPU.
    ctx->api->pinned_memory_size[*ptr] = size;
    // yasai: pinned host memory is mapped in place (device address == host address).
    host_window::add(reinterpret_cast<uint64_t>(*ptr), size);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

// SST malloc done by vanadis, we just need to record the memory addr
cudaError_t CUDARTAPI cudaMallocHostSSTInternal(
    void *addr, size_t size, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // track pinned memory size allocated in the host so that same amount of
  // memory is also allocated in GPU.
  ctx->api->pinned_memory_size[addr] = size;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaMallocPitchInternal(void **devPtr, size_t *pitch, size_t width,
                        size_t height, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned malloc_width_inbytes = width;
  printf("GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(
      malloc_width_inbytes * height);
  pitch[0] = malloc_width_inbytes;
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

cudaError_t cudaHostGetDevicePointerInternal(void **pDevice, void *pHost,
                                             unsigned int flags,
                                             gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: memory registered in host_window is accessed in place by kernels, so its device
  // pointer is the host pointer itself (as with unified virtual addressing).
  if (host_window::contains(reinterpret_cast<uint64_t>(pHost), 1)) {
    *pDevice = pHost;
    return g_last_cudaError = cudaSuccess;
  }
  // only cpu memory allocation happens in cudaHostAlloc. Linking with device
  // pointer to pinned memory happens here.
  // TODO: once kernel is executed, the contents in global pointer of GPU must
  // be copied back to CPU host pointer!
  flags = 0;
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  std::map<void *, size_t>::const_iterator i =
      ctx->api->pinned_memory_size.find(pHost);
  assert(i != ctx->api->pinned_memory_size.end());
  size_t size = i->second;
  *pDevice = gpu->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*pDevice);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*pDevice] = size;
  }
  if (*pDevice) {
    ctx->api->pinned_memory[pHost] = pDevice;
    // Copy contents in cpu to gpu
    gpu->memcpy_to_gpu((size_t)*pDevice, pHost, size);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI cudaMallocArrayInternal(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height __dv(1), gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned size =
      width * height * ((desc->x + desc->y + desc->z + desc->w) / 8);
  CUctx_st *context = GPGPUSim_Context(ctx);
  (*array) = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  (*array)->desc = *desc;
  (*array)->width = width;
  (*array)->height = height;
  (*array)->size = size;
  (*array)->dimensions = 2;
  ((*array)->devPtr32) =
      (int)(long long)context->get_device()->get_gpgpu()->gpu_mallocarray(size);
  printf("GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n",
         ((*array)->devPtr32));
  ((*array)->devPtr) = (void *)(long long)((*array)->devPtr32);
  if (((*array)->devPtr)) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI
cudaMemcpyInternal(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  // gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
  if (kind == cudaMemcpyHostToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation(src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDeviceToHost)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, dst, count, 0));
  else if (kind == cudaMemcpyDeviceToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDefault) {
    // yasai: infer the direction from the simulated heap and the registered host windows
    // (host stack and heap addresses are also above GLOBAL_HEAP_START on x86-64).
    gpgpu_t *gpu = GPGPUSim_Context(ctx)->get_device()->get_gpgpu();
    auto is_dev = [&](const void *p) {
      const unsigned long long a = (unsigned long long)p;
      return (a >= GLOBAL_HEAP_START && a < gpu->dev_malloc_top()) ||
             host_window::contains(a, 1);
    };
    const bool sd = is_dev(src), dd = is_dev(dst);
    if (sd && dd)
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, 0));  // device to device
    else if (sd)
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, 0));  // device to host
    else if (dd)
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(src, (size_t)dst, count, 0));  // host to device
    else
      memcpy(dst, src, count);  // host to host
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = count;
  printf("GPGPU-Sim PTX: cudaMemcpyToArray\n");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)(dst->devPtr), src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst->devPtr, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)(dst->devPtr), (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpyToArray - ERROR : unsupported "
        "cudaMemcpyKind\n");
    abort();
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DInternal(
    void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
    size_t height, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch * height;
  gpgpusim_ptx_assert((dpitch == spitch),
                      "different src and dst pitch not supported yet");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)dst, src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)dst, (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch * height;
  size_t channel_size = dst->desc.w + dst->desc.x + dst->desc.y + dst->desc.z;
  gpgpusim_ptx_assert(
      ((channel_size % 8) == 0),
      "none byte multiple destination channel size not supported (sz=%u)",
      channel_size);
  unsigned elem_size = channel_size / 8;
  gpgpusim_ptx_assert((dst->dimensions == 2),
                      "copy to none 2D array not supported");
  gpgpusim_ptx_assert((wOffset == 0), "non-zero wOffset not yet supported");
  gpgpusim_ptx_assert((hOffset == 0), "non-zero hOffset not yet supported");
  gpgpusim_ptx_assert((dst->height == (int)height),
                      "partial copy not supported");
  gpgpusim_ptx_assert((elem_size * dst->width == width),
                      "partial copy not supported");
  gpgpusim_ptx_assert((spitch == width), "spitch != width not supported");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)(dst->devPtr), src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst->devPtr, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)dst->devPtr, (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolInternal(
    const char *symbol, const void *src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  assert(kind == cudaMemcpyHostToDevice);
  printf("GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p\n", symbol);
  // stream_operation( const char *symbol, const void *src, size_t count, size_t
  // offset )
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(src, symbol, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,src,count,offset,1,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolInternal(
    void *dst, const char *symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  assert(kind == cudaMemcpyDeviceToHost);
  printf("GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(symbol, dst, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyAsyncInternal(
    void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  // yasai: a host source that is not pinned (not a registered host window) is pageable
  // memory. CUDA copies pageable data before cudaMemcpyAsync returns, so the caller may reuse
  // the buffer at once (NCCL passes stack variables), while the stream operation here reads the
  // source only when it executes. The data are therefore copied into a snapshot buffer now.
  // Snapshots are kept until process exit (NCCL's are small; larger copies are rare here).
  auto snapshot = [&](const void *p) -> const void * {
    if (host_window::contains((unsigned long long)p, 1)) return p;
    static std::vector<std::unique_ptr<unsigned char[]>> keep;
    keep.emplace_back(new unsigned char[count ? count : 1]);
    memcpy(keep.back().get(), p, count);
    return keep.back().get();
  };
  if (kind == cudaMemcpyDefault) {
    gpgpu_t *gpu = GPGPUSim_Context(ctx)->get_device()->get_gpgpu();
    auto is_dev = [&](const void *p) {
      const unsigned long long a = (unsigned long long)p;
      return (a >= GLOBAL_HEAP_START && a < gpu->dev_malloc_top()) ||
             host_window::contains(a, 1);
    };
    if (!is_dev(src) && is_dev(dst)) kind = cudaMemcpyHostToDevice;
  }
  switch (kind) {
    case cudaMemcpyHostToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(snapshot(src), (size_t)dst, count, s));
      break;
    case cudaMemcpyDeviceToHost:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, s));
      break;
    case cudaMemcpyDeviceToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, s));
      break;
    case cudaMemcpyDefault: {
      // yasai: infer the direction. An address is device memory when it lies in the
      // simulated heap [GLOBAL_HEAP_START, top) or in a registered host window (which kernels
      // and device copies access in place); other addresses are host memory.
      gpgpu_t *gpu = GPGPUSim_Context(ctx)->get_device()->get_gpgpu();
      auto is_dev = [&](const void *p) {
        const unsigned long long a = (unsigned long long)p;
        return (a >= GLOBAL_HEAP_START && a < gpu->dev_malloc_top()) ||
               host_window::contains(a, 1);
      };
      const bool sd = is_dev(src), dd = is_dev(dst);
      if (sd && dd)
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation((size_t)src, (size_t)dst, count, s));
      else if (sd)
        ctx->the_gpgpusim->g_stream_manager->push(stream_operation((size_t)src, dst, count, s));
      else if (dd)
        ctx->the_gpgpusim->g_stream_manager->push(stream_operation(src, (size_t)dst, count, s));
      else
        memcpy(dst, src, count);  // host to host
      break;
    }
    default:
      abort();
  }
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 8000)
cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  printf(
      "GPGPU-Sim PTX: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags "
      "%p\n",
      hostFunc);
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFunc);
  printf(
      "Calculate Maxium Active Block with function ptr=%p, blockSize=%d, "
      "SMemSize=%lu\n",
      hostFunc, blockSize, dynamicSMemSize);
  // yasai: cudaOccupancyDisableCachingOverride only changes how caching is assumed, not the
  // resource limits the simulator counts, so both flags give the same count (CUTLASS 2.x
  // GemmUniversal asks with that flag); the dynamic shared memory of the query is counted.
  if (flags == cudaOccupancyDefault || flags == cudaOccupancyDisableCachingOverride) {
    if (entry == NULL) {
      *numBlocks = 0;
      printf("GPGPU-Sim PTX: cudaOccupancyMaxActiveBlocksPerMultiprocessor: no PTX for %s\n",
             context->yasai_pending_name(hostFunc));
      return g_last_cudaError = cudaErrorInvalidDeviceFunction;
    }
    // create kernel_info based on entry
    dim3 gridDim(context->get_device()->get_gpgpu()->max_cta_per_core() *
                 context->get_device()->get_gpgpu()->get_config().num_shader());
    dim3 blockDim(blockSize);
    // because this fuction is only checking for resource requirements, we do
    // not care which stream this kernel runs at, just picked -1
    kernel_info_t result(gridDim, blockDim, entry, -1);
    result.set_dynamic_smem(unsigned(dynamicSMemSize));
    // if(entry == NULL){
    //	*numBlocks = 1;
    //	return g_last_cudaError = cudaErrorUnknown;
    //}
    *numBlocks = context->get_device()->get_gpgpu()->get_max_cta(result);
    printf("Maximum size is %d with gridDim %d and blockDim %d\n", *numBlocks,
           gridDim.x, blockDim.x);
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

#endif

__host__ cudaError_t CUDARTAPI cudaMemsetInternal(
    void *mem, int c, size_t count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI
cudaMemsetAsyncInternal(void *mem, int c, size_t count, cudaStream_t stream = 0,
                        gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: WARNING: Asynchronous memset not supported (%s)\n",
         __my_func__);
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaGLMapBufferObjectInternal(void **devPtr, GLuint bufferObj,
                                          gpgpu_context *gpgpu_ctx = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#ifdef OPENGL_SUPPORT
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  GLint buffer_size = 0;
  CUctx_st *context = GPGPUSim_Context(ctx);

  glbmap_entry_t *p = ctx->api->g_glbmap;
  while (p && p->m_bufferObj != bufferObj) p = p->m_next;
  if (p == NULL) {
    glBindBuffer(GL_ARRAY_BUFFER, bufferObj);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &buffer_size);
    assert(buffer_size != 0);
    *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(buffer_size);

    // create entry and insert to front of list
    glbmap_entry_t *n = (glbmap_entry_t *)calloc(1, sizeof(glbmap_entry_t));
    n->m_next = ctx->api->g_glbmap;
    ctx->api->g_glbmap = n;

    // initialize entry
    n->m_bufferObj = bufferObj;
    n->m_devPtr = *devPtr;
    n->m_size = buffer_size;

    p = n;
  } else {
    buffer_size = p->m_size;
    *devPtr = p->m_devPtr;
  }

  if (*devPtr) {
    char *data = (char *)calloc(p->m_size, 1);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, data);
    memcpy_to_gpu((size_t)*devPtr, data, buffer_size);
    free(data);
    printf(
        "GPGPU-Sim PTX: cudaGLMapBufferObject %zu bytes starting at 0x%llx..\n",
        (size_t)buffer_size, (unsigned long long)*devPtr);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }

  return g_last_cudaError = cudaSuccess;
#else
  fflush(stdout);
  fflush(stderr);
  printf(
      "GPGPU-Sim PTX: GPGPU-Sim support for OpenGL integration disabled -- "
      "exiting\n");
  fflush(stdout);
  exit(50);
#endif
}

#if CUDART_VERSION >= 6050
CUresult cuLinkAddFileInternal(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  static bool addedFile = false;
  if (addedFile) {
    printf(
        "GPGPU-Sim PTX: ERROR: cuLinkAddFile does not support multiple "
        "files\n");
    abort();
  }

  // blocking
  assert(type == CU_JIT_INPUT_PTX);
  CUctx_st *context = GPGPUSim_Context(ctx);
  char *file = getenv("PTX_JIT_PATH");
  if (file == NULL) {
    printf("GPGPU-Sim PTX: ERROR: PTX_JIT_PATH has not been set\n");
    abort();
  }
  strcat(file, "/");
  strcat(file, path);
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(file);
  std::string fname(path);
  ctx->api->name_symtab[fname] = symtab;
  context->add_binary(symtab, 1);
  ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT,
                                context->get_device()->get_gpgpu());
  ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                           context->get_device()->get_gpgpu());
  addedFile = true;
  return CUDA_SUCCESS;
}
#endif

#if (CUDART_VERSION >= 2010)

cudaError_t cudaHostAllocInternal(void **pHost, size_t bytes,
                                  unsigned int flags,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *pHost = malloc(bytes);
  // need to track the size allocated so that cudaHostGetDevicePointer() can
  // function properly.
  // TODO: vary this function behavior based on flags value (following nvidia
  // documentation)
  ctx->api->pinned_memory_size[*pHost] = bytes;
  // yasai: pinned host memory is mapped in place (device address == host address).
  if (*pHost) host_window::add(reinterpret_cast<uint64_t>(*pHost), bytes);
  if (*pHost)
    return g_last_cudaError = cudaSuccess;
  else
    return g_last_cudaError = cudaErrorMemoryAllocation;
}

#endif

size_t getMaxThreadsPerBlock(struct cudaFuncAttributes *attr,
                             gpgpu_context *ctx) {
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  struct cudaDeviceProp prop;

  prop = *dev->get_prop();

  size_t max = prop.maxThreadsPerBlock;

  if (attr->numRegs && (size_t)(prop.regsPerBlock / attr->numRegs) < max)
    max = prop.regsPerBlock / attr->numRegs;

  if (attr->sharedSizeBytes &&
      (prop.sharedMemPerBlock / attr->sharedSizeBytes) < max)
    max = prop.sharedMemPerBlock / attr->sharedSizeBytes;

  return max;
}

cudaError_t CUDARTAPI cudaFuncGetAttributesInternal(
    struct cudaFuncAttributes *attr, const char *hostFun,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  if (entry) {
    const struct gpgpu_ptx_sim_info *kinfo = entry->get_kernel_info();
    attr->sharedSizeBytes = kinfo->smem;
    attr->constSizeBytes = kinfo->cmem;
    attr->localSizeBytes = kinfo->lmem;
    attr->numRegs = kinfo->regs;
    if (kinfo->maxthreads > 0)
      attr->maxThreadsPerBlock = kinfo->maxthreads;
    else
      attr->maxThreadsPerBlock = getMaxThreadsPerBlock(attr, ctx);
#if CUDART_VERSION >= 3000
    attr->ptxVersion = kinfo->ptx_version;
    attr->binaryVersion = kinfo->sm_target;
#endif
    if (g_debug_execution >= 3)
      printf("GPGPU-Sim PTX: cudaFuncGetAttributes %s: ptxVersion=%d binaryVersion=%d regs=%d\n",
             entry->get_name().c_str(), attr->ptxVersion, attr->binaryVersion, attr->numRegs);
  } else {
    // yasai: a kernel registered without PTX is not a valid device function (NCCL skips it).
    printf("GPGPU-Sim PTX: cudaFuncGetAttributes: no PTX for %s\n",
           context->yasai_pending_name(hostFun));
    return g_last_cudaError = cudaErrorInvalidDeviceFunction;
  }
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION > 5000)
__host__ cudaError_t CUDARTAPI
cudaDeviceGetAttributeInternal(int *value, enum cudaDeviceAttr attr, int device,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  const struct cudaDeviceProp *prop;
  _cuda_device_id *dev = ctx->GPGPUSim_Init();

  if (device <= dev->num_devices()) {
    prop = dev->get_prop();
    switch ((int)attr) {
      case 1:
        *value = prop->maxThreadsPerBlock;
        break;
      case 2:
        *value = prop->maxThreadsDim[0];
        break;
      case 3:
        *value = prop->maxThreadsDim[1];
        break;
      case 4:
        *value = prop->maxThreadsDim[2];
        break;
      case 5:
        *value = prop->maxGridSize[0];
        break;
      case 6:
        *value = prop->maxGridSize[1];
        break;
      case 7:
        *value = prop->maxGridSize[2];
        break;
      case 8:
        *value = prop->sharedMemPerBlock;
        break;
      case 9:
        *value = prop->totalConstMem;
        break;
      case 10:
        *value = prop->warpSize;
        break;
      case 11:
        *value = 16;  // dummy value
        break;
      case 12:
        *value = prop->regsPerBlock;
        break;
      case 13:
#if (CUDART_VERSION >= 13000)
        *value = dev->get_gpgpu()->shader_clock();
#else
        *value = prop->clockRate;
#endif
        printf("GPGPU-Sim: cudaDevAttrClockRate returning %d kHz (%.0f MHz)\n",
               *value, *value / 1000.0);
        break;
      case 14:
        *value = prop->textureAlignment;
        break;
      case 15:
        *value = 0;
        break;
      case 16:
        *value = prop->multiProcessorCount;
        break;
      case 17:
      case 18:
      case 19:
        *value = 0;
        break;
      case 21:
      case 22:
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 42:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 52:
      case 53:
      case 55:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
      case 62:
      case 63:
      case 64:
      case 66:
      case 67:
      case 69:
      case 70:
      case 71:
      case 73:
      case 74:
      case 77:
        *value = 1000;  // dummy value
        break;
      case 29:
      case 43:
      case 54:
      case 65:
      case 68:
      case 72:
        *value = 10;  // dummy value
        break;
      case 30:
      case 51:
        *value = 128;  // dummy value
        break;
      case 31:
        *value = 1;
        break;
      case 32:
        *value = 0;
        break;
      case 33:
      case 50:
        *value = 0;  // dummy value
        break;
      case 34:
        *value = 0;
        break;
      case 35:
        *value = 0;
        break;
      case 36:
        *value = 1250000;  // CK value for 1080ti
        break;
      case 37:
        *value = 352;  // value for 1080ti
        break;
      case 38:
        *value = 3000000;  // value for 1080ti
        break;
      case 39:
        *value = dev->get_gpgpu()->threads_per_core();
        break;
      case 40:
        *value = 0;
        break;
      case 41:
        *value = 0;
        break;
      case 75:  // cudaDevAttrComputeCapabilityMajor
        *value = prop->major;
        break;
      case 76:  // cudaDevAttrComputeCapabilityMinor
        *value = prop->minor;
        break;
      case 78:
        *value = 0;  // TODO: as of now, we dont support stream priorities.
        break;
      case 79:
        *value = 0;
        break;
      case 80:
        *value = 0;
        break;
#if (CUDART_VERSION > 5050)
      case 81:
        *value = prop->sharedMemPerMultiprocessor;
        break;
      case 82:
        *value = prop->regsPerMultiprocessor;
        break;
#endif
      case 83:
      case 84:
      case 85:
      case 86:
        *value = 0;
        break;
      case 87:
        *value = 4;  // dummy value
        break;
      case 88:
      case 89:
        *value = 0;
        break;
      case 90:  // cudaDevAttrComputePreemptionSupported
        *value = 0;
        break;
      case 91:  // cudaDevAttrCanUseHostPointerForRegisteredMem
        *value = 0;
        break;
      case 92:  // cudaDevAttrReserved92
      case 93:  // cudaDevAttrReserved93
      case 94:  // cudaDevAttrReserved94
        *value = 0;
        break;
      case 95:  // cudaDevAttrCooperativeLaunch
        *value = 1;  // Support cooperative kernels
        break;
      case 96:  // cudaDevAttrCooperativeMultiDeviceLaunch (deprecated)
        *value = 0;
        break;
      case 97:  // cudaDevAttrMaxSharedMemoryPerBlockOptin
        *value = dev->get_gpgpu()->shared_mem_per_block_optin();
        break;
      case 98:  // cudaDevAttrCanFlushRemoteWrites
        *value = 0;
        break;
      case 99:  // cudaDevAttrHostRegisterSupported
        *value = 1;
        break;
      case 100:  // cudaDevAttrPageableMemoryAccessUsesHostPageTables
        *value = 0;
        break;
      case 101:  // cudaDevAttrDirectManagedMemAccessFromHost
        *value = 1;
        break;
      case 102:  // yasai: CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED (cuMem*)
      case 103:  // yasai: CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
        *value = 1;
        break;
      case 104:  // Unassigned
      case 105:  // Unassigned
        *value = 0;
        break;
      case 106:  // cudaDevAttrMaxBlocksPerMultiprocessor
        *value = 32;  // Reasonable default for modern GPUs
        break;
      case 107:  // Unassigned
        *value = 0;
        break;
      case 108:  // cudaDevAttrMaxPersistingL2CacheSize
        *value = prop->l2CacheSize;  // Use L2 cache size from device properties
        break;
      case 109:  // cudaDevAttrMaxAccessPolicyWindowSize
        *value = 0;
        break;
      case 110:  // Unassigned
        *value = 0;
        break;
      case 111:  // cudaDevAttrReservedSharedMemoryPerBlock
        *value = 0;
        break;
      case 112:  // cudaDevAttrSparseCudaArraySupported
        *value = 0;
        break;
      case 113:  // cudaDevAttrHostRegisterReadOnlySupported
        *value = 0;
        break;
      case 114:  // cudaDevAttrTimelineSemaphoreInteropSupported
        *value = 0;
        break;
      case 115:  // cudaDevAttrMemoryPoolsSupported
        *value = 0;
        break;
      case 116:  // cudaDevAttrGPUDirectRDMASupported
        *value = 0;
        break;
      case 117:  // cudaDevAttrGPUDirectRDMAFlushWritesOptions
        *value = 0;
        break;
      case 118:  // cudaDevAttrGPUDirectRDMAWritesOrdering
        *value = 0;
        break;
      case 119:  // cudaDevAttrMemoryPoolSupportedHandleTypes
        *value = 0;
        break;
      case 120:  // cudaDevAttrClusterLaunch
        *value = 1;  // Support cluster launch for SM 9.0+
        break;
      case 121:  // cudaDevAttrDeferredMappingCudaArraySupported
        *value = 0;
        break;
      case 122:  // cudaDevAttrReserved122
      case 123:  // cudaDevAttrReserved123
      case 124:  // cudaDevAttrReserved124
        *value = 0;
        break;
      case 125:  // cudaDevAttrIpcEventSupport (yasai: events in shared memory)
        *value = 1;
        break;
      case 126:  // cudaDevAttrMemSyncDomainCount
        *value = 0;
        break;
      case 127:  // cudaDevAttrReserved127
      case 128:  // cudaDevAttrReserved128
      case 129:  // cudaDevAttrReserved129
        *value = 0;
        break;
      case 130:  // cudaDevAttrNumaConfig
        *value = 0;
        break;
      case 131:  // cudaDevAttrNumaId
        *value = 0;
        break;
      case 132:  // yasai: CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED (shared_devmem multicast);
                 // YASAI_MULTICAST=0 turns it off
        *value = (getenv("YASAI_MULTICAST") && atoi(getenv("YASAI_MULTICAST")) == 0) ? 0 : 1;
        break;
      case 133:  // cudaDevAttrMpsEnabled
        *value = 0;
        break;
      case 134:  // cudaDevAttrHostNumaId
        *value = 0;
        break;
      // yasai: the CUDA 13 attributes after 134. The PCI ids are those of an RTX 5090
      // (device 0x2b85, vendor 0x10de; the simulated device has no subsystem id); the CIG, host
      // NUMA, host memory pool and partial host atomic features are absent.
      case 135:  // cudaDevAttrD3D12CigSupported
      case 136:
      case 137:
      case 138:  // cudaDevAttrVulkanCigSupported
        *value = 0;
        break;
      case 139:  // cudaDevAttrGpuPciDeviceId
        *value = int((0x2b85u << 16) | 0x10deu);
        break;
      case 140:  // cudaDevAttrGpuPciSubsystemId
      case 141:  // cudaDevAttrReserved141
      case 142:  // cudaDevAttrHostNumaMemoryPoolsSupported
      case 143:  // cudaDevAttrHostNumaMultinodeIpcSupported
      case 144:  // cudaDevAttrHostMemoryPoolsSupported
      case 145:  // cudaDevAttrReserved145
      case 146:
      case 147:  // cudaDevAttrOnlyPartialHostNativeAtomicSupported
        *value = 0;
        break;
      default:
        printf("ERROR: Attribute number %d unimplemented \n", attr);
        abort();
    }
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}
#endif

__host__ cudaError_t CUDARTAPI cudaBindTextureInternal(
    size_t *offset, const struct textureReference *texref, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX),
    gpgpu_context *gpgpu_ctx = NULL) {
#if (CUDART_VERSION <= 1200)
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf(
      "GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = "
      "%zu\n",
      sizeof(struct textureReference));
  struct cudaArray *array;
  array = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  array->desc = *desc;
  array->size = size;
  array->width = size;
  array->height = 1;
  array->dimensions = 1;
  array->devPtr = (void *)devPtr;
  array->devPtr32 = (int)(long long)devPtr;
  offset = 0;
  printf("GPGPU-Sim PTX:   size = %zu\n", size);
  printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n",
         desc->x, desc->y, desc->z, desc->w);
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
  devPtr = (void *)(long long)array->devPtr32;
  printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArrayInternal(
    const struct textureReference *texref, const struct cudaArray *array,
    const struct cudaChannelFormatDesc *desc, gpgpu_context *gpgpu_ctx = NULL) {
#if (CUDART_VERSION <= 1200)
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaUnbindTextureInternal(
    const struct textureReference *texref, gpgpu_context *gpgpu_ctx = NULL) {
#if (CUDART_VERSION <= 1200)
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf(
      "GPGPU-Sim PTX: in cudaUnbindTexture: sizeof(struct textureReference) = "
      "%zu\n",
      sizeof(struct textureReference));
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));

  gpu->gpgpu_ptx_sim_unbindTexture(texref);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernelInternal(
    const char *hostFun, dim3 gridDim, dim3 blockDim, const void **args,
    size_t sharedMem, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }

  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
    // Get the actual shared memory size from the kernel configuration
    // instead of using the potentially garbage sharedMem parameter
    if (!ctx->api->g_cuda_launch_stack.empty()) {
      kernel_config config = ctx->api->g_cuda_launch_stack.back();
      printf("shared mem size = %zu\n", config.shared_mem());
    } else {
      printf("shared mem size = %zu (no config available)\n", sharedMem);
    }
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
#if CUDART_VERSION < 10000
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream, ctx);
#endif
  for (unsigned i = 0; i < entry->num_args(); i++) {
    std::pair<size_t, unsigned> p = entry->get_param_config(i);
    cudaSetupArgumentInternal(args[i], p.first, p.second);
  }

  return cudaLaunchInternal(hostFun, ctx);
}

__host__ cudaError_t CUDARTAPI cudaStreamCreateInternal(
    cudaStream_t *stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: cudaStreamCreate\n");
#if (CUDART_VERSION >= 3000)
  *stream = new struct CUstream_st();
  ctx->the_gpgpusim->g_stream_manager->add_stream(*stream);
#else
  *stream = 0;
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroyInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  // per-stream synchronization required for application using external
  // libraries without explicit synchronization in the code to avoid the
  // stream_manager from spinning forever to destroy non-empty streams without
  // making any forward progress.
  stream->synchronize();
  ctx->the_gpgpusim->g_stream_manager->destroy_stream(stream);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronizeInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  if (stream == NULL) ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
  stream->synchronize();
#else
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
#endif
  return g_last_cudaError = cudaSuccess;
}

void __cudaRegisterTextureInternal(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext,
    gpgpu_context *gpgpu_ctx =
        NULL)  // passes in a newly created textureReference
{
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  std::string devStr(deviceName);
#if (CUDART_VERSION > 4020)
  if (devStr.size() > 2 && devStr.data()[0] == ':' && devStr.data()[1] == ':')
    devStr = devStr.replace(0, 2, "");
#endif
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in __cudaRegisterTexture:\n");
  gpu->gpgpu_ptx_sim_bindNameToTexture(devStr.data(), hostVar, dim, norm, ext);
  printf("GPGPU-Sim PTX:   int dim = %d\n", dim);
  printf("GPGPU-Sim PTX:   int norm = %d\n", norm);
  printf("GPGPU-Sim PTX:   int ext = %d\n", ext);
  printf(
      "GPGPU-Sim PTX:   Execution warning: Not finished implementing \"%s\"\n",
      __my_func__);
}

cudaError_t cudaGLUnmapBufferObjectInternal(GLuint bufferObj,
                                            gpgpu_context *gpgpu_ctx = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#ifdef OPENGL_SUPPORT
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  CUctx_st *ctx = GPGPUSim_Context(ctx);
  glbmap_entry_t *p = ctx->api->g_glbmap;
  while (p && p->m_bufferObj != bufferObj) p = p->m_next;
  if (p == NULL) return g_last_cudaError = cudaErrorUnknown;

  char *data = (char *)calloc(p->m_size, 1);
  memcpy_from_gpu(data, (size_t)p->m_devPtr, p->m_size);
  glBufferSubData(GL_ARRAY_BUFFER, 0, p->m_size, data);
  free(data);

  return g_last_cudaError = cudaSuccess;
#else
  fflush(stdout);
  fflush(stderr);
  printf("GPGPU-Sim PTX: support for OpenGL integration disabled -- exiting\n");
  fflush(stdout);
  exit(50);
#endif
}

#if CUDART_VERSION >= 3000

__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfigInternal(const char *func, enum cudaFuncCache cacheConfig,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  context->get_device()->get_gpgpu()->set_cache_config(
      context->get_kernel(func)->get_name(), (FuncCache)cacheConfig);
  return g_last_cudaError = cudaSuccess;
}

#endif

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuLaunchKernelInternal(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  const char *hostFun = (const char *)f;
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  cudaConfigureCallInternal(dim3(gridDimX, gridDimY, gridDimZ),
                            dim3(blockDimX, blockDimY, blockDimZ),
                            sharedMemBytes, (cudaStream_t)hStream, ctx);
  // yasai: copies of the parameters taken from an extra buffer. kernel_config::set_arg keeps
  // only the pointer, and the parameters are read during cudaLaunchInternal, so the copies must
  // live until the launch returns (they were loop locals, and the kernel then read freed memory).
  std::vector<std::vector<unsigned char>> extra_args;
  if (extra != NULL) {
    // yasai: arguments packed in one buffer (CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE), as NCCL
    // passes them. Each parameter is taken from its offset in the buffer; bytes past the end of
    // the buffer are zero.
    const unsigned char *buf = NULL;
    size_t buf_size = 0;
    for (unsigned k = 0; extra[k] != CU_LAUNCH_PARAM_END; k += 2) {
      if (extra[k] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = (const unsigned char *)extra[k + 1];
      else if (extra[k] == CU_LAUNCH_PARAM_BUFFER_SIZE) buf_size = *(size_t *)extra[k + 1];
    }
    if (buf == NULL) return CUDA_ERROR_INVALID_VALUE;
    extra_args.reserve(entry->num_args());
    for (unsigned i = 0; i < entry->num_args(); i++) {
      std::pair<size_t, unsigned> p = entry->get_param_config(i);
      extra_args.emplace_back(p.first, 0);
      std::vector<unsigned char> &arg = extra_args.back();
      if (p.second < buf_size) {
        size_t n = std::min(p.first, buf_size - p.second);
        memcpy(arg.data(), buf + p.second, n);
      }
      cudaSetupArgumentInternal(arg.data(), p.first, p.second, ctx);
    }
  } else {
    for (unsigned i = 0; i < entry->num_args(); i++) {
      std::pair<size_t, unsigned> p = entry->get_param_config(i);
      cudaSetupArgumentInternal(kernelParams[i], p.first, p.second, ctx);
    }
  }
  cudaError_t launch_result = cudaLaunchInternal(hostFun, ctx);
  return launch_result == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
#endif /* CUDART_VERSION >= 4000 */

CUevent_st *get_event(cudaEvent_t event) {
  unsigned event_uid;
#if CUDART_VERSION >= 3000
  event_uid = event->get_uid();
#else
  event_uid = event;
#endif
  event_tracker_t::iterator e = g_timer_events.find(event_uid);
  if (e == g_timer_events.end()) return NULL;
  return e->second;
}

__host__ cudaError_t CUDARTAPI cudaEventRecordInternal(
    cudaEvent_t event, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (!e) return g_last_cudaError = cudaErrorUnknown;
  struct CUstream_st *s = (struct CUstream_st *)stream;
  stream_operation op(e, s);
  e->issue();
  ctx->the_gpgpusim->g_stream_manager->push(op);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEventInternal(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // reference:
  // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__STREAM_gfe68d207dc965685d92d3f03d77b0876.html
  CUevent_st *e = get_event(event);
  if (!e) {
    printf(
        "GPGPU-Sim API: Error at cudaStreamWaitEvent. Event is not created "
        ".\n");
    return g_last_cudaError = cudaErrorInvalidResourceHandle;
  } else if (e->num_issued() == 0) {
    printf(
        "GPGPU-Sim API: Warning: cudaEventRecord has not been called on event "
        "before calling cudaStreamWaitEvent.\nNothin    g to be done.\n");
    return g_last_cudaError = cudaSuccess;
  }
  if (!stream) {
    ctx->the_gpgpusim->g_stream_manager->pushCudaStreamWaitEventToAllStreams(
        e, flags);
  } else {
    struct CUstream_st *s = (struct CUstream_st *)stream;
    stream_operation op(s, e, flags);
    ctx->the_gpgpusim->g_stream_manager->push(op);
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadExitInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  ctx->exit_simulation();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Called on host side
  ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI
cudaDeviceSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Blocks until the device has completed all preceding requested tasks
  ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*******************************************************************************
 *                                                                              *
 *   SST Specific functions, used by Balar *
 *                                                                              *
 *******************************************************************************/

/**
 * @brief Custom function to get CUDA function parameter size and offset
 *        from PTX parsing result
 *
 * @param hostFun
 * @param index
 * @return std::tuple<cudaError_t, size_t, unsigned>
 */
std::tuple<cudaError_t, size_t, unsigned> SST_cudaGetParamConfig(
    uint64_t hostFun, unsigned index) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  function_info *entry = context->get_kernel((char *)hostFun);
  cudaError_t result = cudaSuccess;
  size_t size = 0;
  unsigned alignment = 0;
  if (index >= entry->num_args()) {
    result = cudaErrorAssert;
  } else {
    std::pair<size_t, unsigned> p = entry->get_param_config(index);
    size = p.first;
    alignment = p.second;
  }
  return std::tuple<cudaError_t, size_t, unsigned>(result, size, alignment);
}

extern "C" {
void SST_receive_mem_reply(unsigned core_id, void *mem_req) {
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  static_cast<sst_gpgpu_sim *>(context->get_device()->get_gpgpu())
      ->SST_receive_mem_reply(core_id, mem_req);
}

bool SST_gpu_core_cycle() { return SST_Cycle(); }

void SST_gpgpusim_numcores_equal_check(unsigned sst_numcores) {
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  static_cast<sst_gpgpu_sim *>(context->get_device()->get_gpgpu())
      ->SST_gpgpusim_numcores_equal_check(sst_numcores);
}

uint64_t cudaMallocSST(void **devPtr, size_t size) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  void *test_malloc;
  test_malloc = (void *)malloc(size);
  void **test_malloc2 = &test_malloc;
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  *test_malloc2 = context->get_device()->get_gpgpu()->gpu_malloc(size);
  printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n", size,
         (unsigned long long)*test_malloc2);
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*test_malloc2);
  return (uint64_t)*test_malloc2;
}

__host__ cudaError_t CUDARTAPI cudaMallocHostSST(void *addr, size_t size) {
  return cudaMallocHostSSTInternal(addr, size);
}

cudaError_t cudaPeekAtLastError(void) { return g_last_cudaError; }

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  return cudaMallocInternal(devPtr, size);
}

__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
  return cudaMallocHostInternal(ptr, size);
}
__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch,
                                               size_t width, size_t height) {
  return cudaMallocPitchInternal(devPtr, pitch, width, height);
}

__host__ cudaError_t CUDARTAPI cudaMallocArray(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height __dv(1)) {
  return cudaMallocArrayInternal(array, desc, width, height);
}

// yasai: managed allocations (cudaMallocManaged), which cudaFree releases as host memory.
static std::set<void *> g_yasai_managed;
// yasai: guards g_yasai_managed.
static std::mutex g_yasai_managed_mutex;

__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  {
    std::lock_guard<std::mutex> lock(g_yasai_managed_mutex);
    if (g_yasai_managed.erase(devPtr)) {
      host_window::remove(reinterpret_cast<uint64_t>(devPtr));
      free(devPtr);
      return g_last_cudaError = cudaSuccess;
    }
  }
  // yasai: unmaps the allocation's segment (device addresses are not reused).
  yasai_vmm::free_device(reinterpret_cast<uint64_t>(devPtr));
  return g_last_cudaError = cudaSuccess;
}
__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  host_window::remove(reinterpret_cast<uint64_t>(ptr));
  free(ptr);  // this will crash the system if called twice
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
};

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

// yasai: the fakeenv recorder (yasai_fakeenv_rec.h) records the calls below when
// YASAI_FAKEENV_OUT is set: an api record and the device operation with the same correlation id.
namespace {
/// Records a copy call and its memcpy operation.
void yasai_rec_copy(const char *api, uint64_t t0, uint32_t corr, cudaError_t ret, size_t count,
                    int kind, const void *stream) {
  yasai_rec::Recorder &r = yasai_rec::Recorder::get();
  const uint64_t t1 = yasai_rec::mono_ns();
  r.api(api, t0, t1, corr, int(ret));
  r.write("\"rec\":\"memcpy\",\"start\":%llu,\"end\":%llu,\"dev\":0,\"ctx\":1,\"stream\":%u,"
          "\"corr\":%u,\"bytes\":%zu,\"copy_kind\":%u,\"src_kind\":0,\"dst_kind\":0,\"graph_node\":0",
          (unsigned long long)t0, (unsigned long long)t1, r.stream_id(stream), corr, count,
          yasai_rec::copy_kind(kind));
}

/// Records a memset call and its memset operation.
void yasai_rec_memset(const char *api, uint64_t t0, uint32_t corr, cudaError_t ret,
                      size_t count, int value, const void *stream) {
  yasai_rec::Recorder &r = yasai_rec::Recorder::get();
  const uint64_t t1 = yasai_rec::mono_ns();
  r.api(api, t0, t1, corr, int(ret));
  r.write("\"rec\":\"memset\",\"start\":%llu,\"end\":%llu,\"dev\":0,\"ctx\":1,\"stream\":%u,"
          "\"corr\":%u,\"bytes\":%zu,\"value\":%u,\"graph_node\":0",
          (unsigned long long)t0, (unsigned long long)t1, r.stream_id(stream), corr, count,
          unsigned(value) & 0xff);
}

/// Records a synchronization call (type: yasai_rec::SyncType).
void yasai_rec_sync(const char *api, uint64_t t0, uint32_t corr, cudaError_t ret, int type,
                    const void *stream, uint64_t event, uint64_t sync_id) {
  yasai_rec::Recorder &r = yasai_rec::Recorder::get();
  const uint64_t t1 = yasai_rec::mono_ns();
  r.api(api, t0, t1, corr, int(ret));
  r.write("\"rec\":\"sync\",\"start\":%llu,\"end\":%llu,\"type\":%d,\"ctx\":1,\"stream\":%u,"
          "\"corr\":%u,\"event\":%llu,\"sync_id\":%llu",
          (unsigned long long)t0, (unsigned long long)t1, type, r.stream_id(stream), corr,
          (unsigned long long)event, (unsigned long long)sync_id);
}
}  // namespace

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
                                          size_t count,
                                          enum cudaMemcpyKind kind) {
  if (!yasai_rec::Recorder::get().on()) return cudaMemcpyInternal(dst, src, count, kind);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = yasai_rec::Recorder::get().corr();
  const cudaError_t ret = cudaMemcpyInternal(dst, src, count, kind);
  yasai_rec_copy("cudaMemcpy", t0, corr, ret, count, int(kind), nullptr);
  return ret;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst,
                                                 size_t wOffset, size_t hOffset,
                                                 const void *src, size_t count,
                                                 enum cudaMemcpyKind kind) {
  return cudaMemcpyToArrayInternal(dst, wOffset, hOffset, src, count, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst,
                                                   const struct cudaArray *src,
                                                   size_t wOffset,
                                                   size_t hOffset, size_t count,
                                                   enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(
    struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch,
                                            const void *src, size_t spitch,
                                            size_t width, size_t height,
                                            enum cudaMemcpyKind kind) {
  return cudaMemcpy2DInternal(dst, dpitch, src, spitch, width, height, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  return cudaMemcpy2DToArrayInternal(dst, wOffset, hOffset, src, spitch, width,
                                     height, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(
    void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(
    struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t width, size_t height,
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(
    const char *symbol, const void *src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
  return cudaMemcpyToSymbolInternal(symbol, src, count, offset, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(
    void *dst, const char *symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
  return cudaMemcpyFromSymbolInternal(dst, symbol, count, offset, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // placeholder; should interact with cudaMalloc and cudaFree?
  *free = 10000000000;
  *total = 10000000000;

  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                               size_t count,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  if (!yasai_rec::Recorder::get().on())
    return cudaMemcpyAsyncInternal(dst, src, count, kind, stream);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = yasai_rec::Recorder::get().corr();
  const cudaError_t ret = cudaMemcpyAsyncInternal(dst, src, count, kind, stream);
  yasai_rec_copy("cudaMemcpyAsync", t0, corr, ret, count, int(kind), stream);
  return ret;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(
    void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                                 const void *src, size_t spitch,
                                                 size_t width, size_t height,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

#if (CUDART_VERSION >= 8000)
cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
      numBlocks, hostFunc, blockSize, dynamicSMemSize, flags);
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count) {
  if (!yasai_rec::Recorder::get().on()) return cudaMemsetInternal(mem, c, count);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = yasai_rec::Recorder::get().corr();
  const cudaError_t ret = cudaMemsetInternal(mem, c, count);
  yasai_rec_memset("cudaMemset", t0, corr, ret, count, c, nullptr);
  return ret;
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *mem, int c, size_t count,
                                               cudaStream_t stream = 0) {
  if (!yasai_rec::Recorder::get().on()) return cudaMemsetAsyncInternal(mem, c, count, stream = 0);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = yasai_rec::Recorder::get().corr();
  const void *recorded_stream = stream;
  const cudaError_t ret = cudaMemsetAsyncInternal(mem, c, count, stream = 0);
  yasai_rec_memset("cudaMemsetAsync", t0, corr, ret, count, c, recorded_stream);
  return ret;
}

__host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c,
                                            size_t width, size_t height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr,
                                                    const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size,
                                                 const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  return cudaGetDeviceCountInternal(count);
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  return cudaGetDevicePropertiesInternal(prop, device);
}

#if (CUDART_VERSION > 5000)
__host__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value,
                                                      enum cudaDeviceAttr attr,
                                                      int device) {
  return cudaDeviceGetAttributeInternal(value, attr, device);
}
#endif

__host__ cudaError_t CUDARTAPI
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
  return cudaChooseDeviceInternal(device, prop);
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  return cudaSetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  return cudaGetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue,
                                                  cudaLimit limit) {
  return cudaDeviceGetLimitInternal(pValue, limit);
}

__host__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream,
                                                     int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len,
                                                     int device) {
  // yasai: a stable made-up bus id per simulated device, bus = YASAI_PCI_BUS_BASE (default
  // 0xd0) + device. The base must not be a bus of the host, because NCCL reads the device's
  // sysfs entry when one exists (bus 1 is a real controller on the lab hosts).
  if (pciBusId == NULL || len <= 0) return g_last_cudaError = cudaErrorInvalidValue;
  const char *base_env = getenv("YASAI_PCI_BUS_BASE");
  const unsigned base = base_env ? unsigned(strtoul(base_env, NULL, 0)) : 0xd0u;
  snprintf(pciBusId, len, "0000:%02x:00.0", (base + device) & 0xff);
  return g_last_cudaError = cudaSuccess;
}

// yasai: inverse of cudaDeviceGetPCIBusId above (device = bus - YASAI_PCI_BUS_BASE).
__host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
  unsigned dom = 0, bus = 0, dev = 0, fn = 0;
  const char *base_env = getenv("YASAI_PCI_BUS_BASE");
  const unsigned base = base_env ? unsigned(strtoul(base_env, NULL, 0)) : 0xd0u;
  if (device == NULL || pciBusId == NULL ||
      sscanf(pciBusId, "%x:%x:%x.%x", &dom, &bus, &dev, &fn) != 4 || bus < base)
    return g_last_cudaError = cudaErrorInvalidValue;
  *device = int(bus - base);
  return g_last_cudaError = cudaSuccess;
}

// yasai: graph construction calls that NCCL uses only while a stream is being captured; the
// simulator never reports capturing (cudaStreamIsCapturing), so these are not reached.
__host__ cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t, const cudaGraphNode_t *,
                                                        const cudaGraphNode_t *,
                                                        const cudaGraphEdgeData *, size_t) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: see cudaGraphAddDependencies.
__host__ cudaError_t CUDARTAPI cudaGraphAddEventRecordNode(cudaGraphNode_t *, cudaGraph_t,
                                                           const cudaGraphNode_t *, size_t,
                                                           cudaEvent_t) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: see cudaGraphAddDependencies.
__host__ cudaError_t CUDARTAPI cudaGraphRetainUserObject(cudaGraph_t, cudaUserObject_t,
                                                         unsigned int, unsigned int) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: see cudaGraphAddDependencies.
__host__ cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies(cudaStream_t,
                                                                   cudaGraphNode_t *,
                                                                   const cudaGraphEdgeData *,
                                                                   size_t, unsigned int) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: memory pools are not modeled; a pool is a token and allocations from it are ordinary
// cudaMalloc allocations.
__host__ cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t *memPool,
                                                 const struct cudaMemPoolProps *) {
  static uintptr_t next_pool = 1;
  if (memPool == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  *memPool = reinterpret_cast<cudaMemPool_t>(next_pool++);
  return g_last_cudaError = cudaSuccess;
}

// yasai: see cudaMemPoolCreate.
__host__ cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t) {
  return g_last_cudaError = cudaSuccess;
}

// yasai: see cudaMemPoolCreate.
__host__ cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t,
                                                       cudaStream_t) {
  return cudaMalloc(ptr, size);
}

// yasai: a batch of copies is issued as one cudaMemcpyAsync per entry with cudaMemcpyDefault.
__host__ cudaError_t CUDARTAPI cudaMemcpyBatchAsync(void *const *dsts, const void *const *srcs,
                                                   const size_t *sizes, size_t count,
                                                   struct cudaMemcpyAttributes *, size_t *,
                                                   size_t, cudaStream_t stream) {
  for (size_t i = 0; i < count; i++) {
    cudaError_t e = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream);
    if (e != cudaSuccess) return g_last_cudaError = e;
  }
  return g_last_cudaError = cudaSuccess;
}

// yasai: runtime functions that PyTorch 2.14 links. Graph memory is not modeled (attributes
// read as 0 and writes are accepted); graph construction answers not supported because the
// simulator never captures; an event record with flags is an ordinary event record.
__host__ cudaError_t CUDARTAPI cudaDeviceGetGraphMemAttribute(int, enum cudaGraphMemAttributeType,
                                                               void *value) {
  if (value) *static_cast<unsigned long long *>(value) = 0;
  return g_last_cudaError = cudaSuccess;
}

// yasai: see cudaDeviceGetGraphMemAttribute.
__host__ cudaError_t CUDARTAPI cudaDeviceSetGraphMemAttribute(int, enum cudaGraphMemAttributeType,
                                                               void *) {
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream);

// yasai: see cudaDeviceGetGraphMemAttribute.
__host__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                        unsigned int) {
  return cudaEventRecord(event, stream);
}

// yasai: see cudaDeviceGetGraphMemAttribute.
__host__ cudaError_t CUDARTAPI cudaGraphAddNode(cudaGraphNode_t *, cudaGraph_t,
                                                const cudaGraphNode_t *, const cudaGraphEdgeData *,
                                                size_t, struct cudaGraphNodeParams *) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: see cudaDeviceGetGraphMemAttribute.
__host__ cudaError_t CUDARTAPI cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle *,
                                                                cudaGraph_t, unsigned int,
                                                                unsigned int) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: see cudaDeviceGetGraphMemAttribute.
__host__ cudaError_t CUDARTAPI cudaStreamBeginCaptureToGraph(cudaStream_t, cudaGraph_t,
                                                             const cudaGraphNode_t *,
                                                             const cudaGraphEdgeData *, size_t,
                                                             enum cudaStreamCaptureMode) {
  return g_last_cudaError = cudaErrorNotSupported;
}

// yasai: see cudaDeviceGetGraphMemAttribute.
__host__ cudaError_t CUDARTAPI cudaUserObjectRelease(cudaUserObject_t, unsigned int) {
  return g_last_cudaError = cudaSuccess;
}

// yasai: legacy IPC (yasai_vmm, shared_devmem): the handle carries the ref of the allocation's
// memfd; the peer maps that segment at a fresh device address of its own.
__host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
                                                   void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (!handle) return g_last_cudaError = cudaErrorInvalidValue;
  if (!yasai_vmm::ipc_get_mem_handle(yasai_current_gpu(), handle,
                                     reinterpret_cast<uint64_t>(devPtr)))
    return g_last_cudaError = cudaErrorInvalidValue;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t cudaIpcOpenMemHandle(void **devPtr,
                                          cudaIpcMemHandle_t handle,
                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)flags;
  uint64_t p = 0;
  if (!devPtr || !yasai_vmm::ipc_open_mem_handle(yasai_current_gpu(), &handle, &p))
    return g_last_cudaError = cudaErrorInvalidValue;
  *devPtr = reinterpret_cast<void *>(p);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t *offset, const struct textureReference *texref, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX)) {
  return cudaBindTextureInternal(offset, texref, devPtr, desc,
                                 size __dv(UINT_MAX));
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array,
    const struct cudaChannelFormatDesc *desc) {
  return cudaBindTextureToArrayInternal(texref, array, desc);
}

__host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const struct textureReference *texref) {
  return cudaUnbindTextureInternal(texref);
}

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
    size_t *offset, const struct textureReference *texref) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(
    const struct textureReference **texref, const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(
    struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *desc = array->desc;
  return g_last_cudaError = cudaSuccess;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

__host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t error = g_last_cudaError;
  g_last_cudaError = cudaSuccess;
  return error;
}

// yasai: names of the error codes the simulator returns or the libraries test for; other codes
// print their number.
static const char *yasai_error_name(cudaError_t error) {
  switch (error) {
#define YASAI_ERR(e) \
  case e:            \
    return #e;
    YASAI_ERR(cudaSuccess)
    YASAI_ERR(cudaErrorInvalidValue)
    YASAI_ERR(cudaErrorMemoryAllocation)
    YASAI_ERR(cudaErrorInitializationError)
    YASAI_ERR(cudaErrorCudartUnloading)
    YASAI_ERR(cudaErrorInvalidConfiguration)
    YASAI_ERR(cudaErrorInvalidPitchValue)
    YASAI_ERR(cudaErrorInvalidSymbol)
    YASAI_ERR(cudaErrorInvalidDevicePointer)
    YASAI_ERR(cudaErrorInvalidMemcpyDirection)
    YASAI_ERR(cudaErrorInsufficientDriver)
    YASAI_ERR(cudaErrorNoDevice)
    YASAI_ERR(cudaErrorInvalidDevice)
    YASAI_ERR(cudaErrorInvalidResourceHandle)
    YASAI_ERR(cudaErrorNotReady)
    YASAI_ERR(cudaErrorInvalidDeviceFunction)
    YASAI_ERR(cudaErrorNotSupported)
    YASAI_ERR(cudaErrorPeerAccessAlreadyEnabled)
    YASAI_ERR(cudaErrorPeerAccessNotEnabled)
    YASAI_ERR(cudaErrorHostMemoryAlreadyRegistered)
    YASAI_ERR(cudaErrorHostMemoryNotRegistered)
    YASAI_ERR(cudaErrorLaunchFailure)
    YASAI_ERR(cudaErrorIllegalAddress)
    YASAI_ERR(cudaErrorStreamCaptureUnsupported)
    YASAI_ERR(cudaErrorUnknown)
#undef YASAI_ERR
    default:
      return nullptr;
  }
}

__host__ const char *cudaGetErrorName(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: implemented (PyTorch formats every CUDA error with it).
  if (const char *n = yasai_error_name(error)) return n;
  static thread_local char buf[64];
  snprintf(buf, sizeof buf, "cudaError_%d", int(error));
  return buf;
}

__host__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: describes the error passed in (the code previously described g_last_cudaError).
  if (error == cudaSuccess) return "no error";
  static thread_local char buf[160];
  const char *n = yasai_error_name(error);
  snprintf(buf, sizeof buf, "GPGPU-Sim: %s (code %d)", n ? n : "error", int(error));
  return buf;
}

// SST specific cuda apis
__host__ cudaError_t CUDARTAPI cudaSetupArgumentSST(uint64_t arg,
                                                    uint8_t value[200],
                                                    size_t size,
                                                    size_t offset) {
  void *local_value;
  local_value = (void *)malloc(size);

  if (arg) {
    memcpy(local_value, (void *)&arg, size);
  } else {
    memcpy(local_value, value, size);
  }
  return cudaSetupArgumentInternal(local_value, size, offset);
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size,
                                                 size_t offset) {
  return cudaSetupArgumentInternal(arg, size, offset);
}

// SST specific cuda apis
__host__ cudaError_t CUDARTAPI cudaLaunchSST(uint64_t hostFun) {
  return cudaLaunchInternal((char *)hostFun);
}

__host__ cudaError_t CUDARTAPI cudaLaunch(const char *hostFun) {
  return cudaLaunchInternal(hostFun);
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const char *hostFun,
                                                dim3 gridDim, dim3 blockDim,
                                                const void **args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
  return cudaLaunchKernelInternal(hostFun, gridDim, blockDim, args, sharedMem,
                                  stream);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
  const cudaError_t ret = cudaStreamCreateInternal(stream);
  if (ret == cudaSuccess && yasai_rec::Recorder::get().on())
    yasai_rec::Recorder::get().stream_id(*stream);  // writes stream_created
  return ret;
}

// TODO: introduce priorities
__host__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(
    cudaStream_t *stream, unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI
cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

__host__ __device__ cudaError_t CUDARTAPI
cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  return cudaStreamDestroyInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  if (!yasai_rec::Recorder::get().on()) return cudaStreamSynchronizeInternal(stream);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = yasai_rec::Recorder::get().corr();
  const cudaError_t ret = cudaStreamSynchronizeInternal(stream);
  yasai_rec_sync("cudaStreamSynchronize", t0, corr, ret, yasai_rec::kStreamSync, stream, 0, 0);
  return ret;
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  if (stream == NULL) return g_last_cudaError = cudaErrorInvalidResourceHandle;
  return g_last_cudaError = stream->empty() ? cudaSuccess : cudaErrorNotReady;
#else
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
  return g_last_cudaError = cudaSuccess;  // it is always success because all
                                          // cuda calls are synchronous
#endif
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = new CUevent_st(false);
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                               cudaStream_t stream) {
  yasai_rec::Recorder &rec = yasai_rec::Recorder::get();
  if (!rec.on()) return cudaEventRecordInternal(event, stream);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = rec.corr();
  const cudaError_t ret = cudaEventRecordInternal(event, stream);
  const uint64_t uid = event ? uint64_t(((CUevent_st *)event)->get_uid()) : 0;
  const uint64_t sync_id = rec.event_record_id(uid);
  rec.api("cudaEventRecord", t0, yasai_rec::mono_ns(), corr, int(ret));
  rec.write("\"rec\":\"event_record\",\"dev\":0,\"ctx\":1,\"stream\":%u,\"corr\":%u,"
            "\"event\":%llu,\"sync_id\":%llu,\"dev_ts\":0",
            rec.stream_id(stream), corr, (unsigned long long)uid, (unsigned long long)sync_id);
  return ret;
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                   cudaEvent_t event,
                                                   unsigned int flags) {
  yasai_rec::Recorder &rec = yasai_rec::Recorder::get();
  if (!rec.on()) return cudaStreamWaitEventInternal(stream, event, flags);
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = rec.corr();
  const cudaError_t ret = cudaStreamWaitEventInternal(stream, event, flags);
  const uint64_t uid = event ? uint64_t(((CUevent_st *)event)->get_uid()) : 0;
  yasai_rec_sync("cudaStreamWaitEvent", t0, corr, ret, yasai_rec::kStreamWaitEvent, stream, uid,
                 rec.event_last_id(uid));
  return ret;
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (e == NULL) {
    return g_last_cudaError = cudaErrorInvalidValue;
  } else if (e->done()) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorNotReady;
  }
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  const bool rec_on = yasai_rec::Recorder::get().on();
  const uint64_t t0 = rec_on ? yasai_rec::mono_ns() : 0;
  const uint32_t corr = rec_on ? yasai_rec::Recorder::get().corr() : 0;
  printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
  fflush(stdout);
  CUevent_st *e = (CUevent_st *)event;
  while (!e->done());
  printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
  fflush(stdout);
  if (rec_on) {
    const uint64_t uid = e ? uint64_t(e->get_uid()) : 0;
    yasai_rec_sync("cudaEventSynchronize", t0, corr, cudaSuccess, yasai_rec::kEventSync, nullptr,
                   uid, yasai_rec::Recorder::get().event_last_id(uid));
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  unsigned event_uid = e->get_uid();
  event_tracker_t::iterator pe = g_timer_events.find(event_uid);
  if (pe == g_timer_events.end())
    return g_last_cudaError = cudaErrorInvalidValue;
  g_timer_events.erase(pe);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms,
                                                    cudaEvent_t start,
                                                    cudaEvent_t end) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  time_t elapsed_time;
  CUevent_st *s = get_event(start);
  CUevent_st *e = get_event(end);
  if (s == NULL || e == NULL) return g_last_cudaError = cudaErrorUnknown;
  elapsed_time = e->clock() - s->clock();
  *ms = 1000 * elapsed_time;
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
  return cudaThreadExitInternal();
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
  return cudaThreadSynchronizeInternal();
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronizeSST(void) {
  // For SST, perform a one-time check and let SST_Cycle()
  // do the polling test and invoke callback to SST
  // to signal ThreadSynchonize done
  gpgpu_context *ctx = GPGPU_Context();
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  // Called on host side
  bool thread_sync_done = ctx->synchronize_check();
  g_last_cudaError = cudaSuccess;
  if (thread_sync_done) {
    // We are already done, so no need to poll for sync done
    ctx->requested_synchronize = false;
    return cudaSuccess;
  } else {
    return cudaErrorNotReady;
  }
}

int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaThreadExit();
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

#if (CUDART_VERSION >= 3010)
int dummy0() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 0;
}

int dummy1() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 2 << 20;
}

typedef int (*ExportedFunction)();

static ExportedFunction exportTable[3] = {&dummy0, &dummy0, &dummy0};

__host__ cudaError_t CUDARTAPI cudaGetExportTable(
    const void **ppExportTable, const cudaUUID_t *pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("cudaGetExportTable: UUID = ");
  for (int s = 0; s < 16; s++) {
    printf("%#2x ", (unsigned char)(pExportTableId->bytes[s]));
  }
  *ppExportTable = &exportTable;

  printf("\n");
  return g_last_cudaError = cudaSuccess;
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

// #include "../../cuobjdump_to_ptxplus/cuobjdump_parser.h"

// extracts all ptx files from binary and dumps into
// prog_name.unique_no.sm_<>.ptx files
void cuda_runtime_api::extract_ptx_files_using_cuobjdump_internal(
    CUctx_st *context, std::string &app_binary) {
  char command[2048];
  char *pytorch_bin = getenv("PYTORCH_BIN");

  char ptx_list_file_name[1024];
  snprintf(ptx_list_file_name, 1024, "_cuobjdump_list_ptx_XXXXXX");
  int fd2 = mkstemp(ptx_list_file_name);
  close(fd2);

  // yasai: PYTORCH_BIN may list several libraries separated by ':' (for example libtorch_cuda.so
  // and the cuBLAS replacement); the PTX files of all of them are listed and extracted, and
  // cuobjdumpParseBinary selects the PTX version per library.
  std::vector<std::string> binaries;
  if (pytorch_bin != NULL && strlen(pytorch_bin) != 0) {
    std::string list(pytorch_bin);
    size_t start = 0;
    while (start <= list.size()) {
      const size_t end = std::min(list.find(':', start), list.size());
      if (end > start) binaries.push_back(list.substr(start, end - start));
      start = end + 1;
    }
  }
  if (binaries.empty()) binaries.push_back(app_binary);
  app_binary = binaries[0];
  app_binary_path = app_binary;

  // only want file names
  std::string list_cmd;
  for (const std::string &b : binaries)
    list_cmd += "$CUDA_INSTALL_PATH/bin/cuobjdump -lptx " + b +
                "  | cut -d \":\" -f 2 | awk '{$1=$1}1' >> " + ptx_list_file_name + "; ";
  snprintf(command, sizeof(command), "%s", list_cmd.c_str());
  if (GPTRACE(PTX_IR)) {
    char cwd_buf[4096];
    [[maybe_unused]] const char *cwd =
        getcwd(cwd_buf, sizeof(cwd_buf)) ? cwd_buf : "<getcwd failed>";
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: extract cwd=%s app_binary=%s list_file=%s\n", cwd,
                 app_binary.c_str(), ptx_list_file_name);
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: extract command=%s\n", command);
  }
  if (yasai_system(command) != 0) {
    printf("WARNING: Failed to execute cuobjdump to get list of ptx files \n");
    exit(0);
  }
  if (!gpgpu_ctx->device_runtime->g_cdp_enabled) {
    // based on the list above, dump ptx files individually. Format of dumped
    // ptx file is prog_name.unique_no.sm_<>.ptx

    // yasai: one `cuobjdump -xptx all` pass writes the same files as one call per listed file.
    // Each call scans the whole binary, so the per-file loop took about 15 s per file on
    // libtorch_cuda.so (418 PTX files, 1.2 GB).
    std::ifstream infile(ptx_list_file_name);
    std::string line;
    while (std::getline(infile, line)) context->no_of_ptx++;
    if (context->no_of_ptx) {
      for (const std::string &b : binaries) {
        printf("Extracting all PTX files of %s (%d listed in total)\n", b.c_str(),
               context->no_of_ptx);
        snprintf(command, sizeof(command),
                 "$CUDA_INSTALL_PATH/bin/cuobjdump -xptx all %s > /dev/null", b.c_str());
        if (yasai_system(command) != 0) {
          printf("ERROR: command: %s failed \n", command);
          exit(0);
        }
      }
    }
  }

  if (!context->no_of_ptx) {
    printf(
        "WARNING: Number of ptx in the executable file are 0. One of the "
        "reasons might be\n");
    printf("\t1. CDP is enabled\n");
    printf("\t2. When using PyTorch, PYTORCH_BIN is not set correctly\n");
  }

  std::ifstream infile(ptx_list_file_name);
  std::string line;
  while (std::getline(infile, line)) {
    // int pos = line.find(std::string(get_app_binary_name(app_binary)));
    size_t pos1 = line.find("sm_");
    size_t pos2 = line.find_last_of(".");
    if (pos1 == std::string::npos && pos2 == std::string::npos) {
      printf("ERROR: PTX list is not in correct format");
      exit(0);
    }
    std::string vstr = line.substr(pos1 + 3, pos2 - pos1 - 3);
    // Extract base version number and suffix
    unsigned version = 0;
    size_t i = 0;
    while (i < vstr.length() && isdigit(vstr[i])) {
      version = version * 10 + (vstr[i] - '0');
      i++;
    }
    if (version_filename.find(version) == version_filename.end()) {
      version_filename[version] = std::set<std::string>();
    }
    version_filename[version].insert(line);
  }
}

void cuda_runtime_api::extract_ptx_files_using_cuobjdump(CUctx_st *context,
                                                         const char *fn) {
  std::string app_binary = get_app_binary(fn);
  this->extract_ptx_files_using_cuobjdump_internal(context, app_binary);
}

void cuda_runtime_api::extract_ptx_files_using_cuobjdump(CUctx_st *context) {
  std::string app_binary = get_app_binary();
  this->extract_ptx_files_using_cuobjdump_internal(context, app_binary);
}

//! Call cuobjdump to extract everything (-elf -sass -ptx)
/*!
 *	This Function extract the whole PTX (for all the files) using cuobjdump
 *	to _cuobjdump_complete_output_XXXXXX then runs a parser to chop it up
 *with each binary in its own file It is also responsible for extracting the
 *libraries linked to the binary if the option is enabled
 * */
void cuda_runtime_api::extract_code_using_cuobjdump_internal(
    CUctx_st *context, std::string &app_binary,
    std::function<void(CUctx_st *)> ctx_extract_ptx_func) {
  // prevent the dumping by cuobjdump everytime we execute the code!
  const char *override_cuobjdump = getenv("CUOBJDUMP_SIM_FILE");
  char command[1000];
  // Running cuobjdump using dynamic link to current process
  snprintf(command, 1000, "md5sum %s ", app_binary.c_str());
  printf("Running md5sum using \"%s\"\n", command);
  if (yasai_system(command)) {
    std::cout << "Failed to execute: " << command << std::endl;
    exit(1);
  }
  // Running cuobjdump using dynamic link to current process
  // Needs the option '-all' to extract PTX from CDP-enabled binary

  // dump ptx for all individial ptx files into sepearte files which is later
  // used by ptxas.
  int result = 0;
#if (CUDART_VERSION >= 6000)
  ctx_extract_ptx_func(context);
  return;
#endif
  // TODO: redundant to dump twice. how can it be prevented?
  // dump only for specific arch
  char fname[1024];
  if ((override_cuobjdump == NULL) || (strlen(override_cuobjdump) == 0)) {
    snprintf(fname, 1024, "_cuobjdump_complete_output_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);
    if (!gpgpu_ctx->device_runtime->g_cdp_enabled)
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass %s > %s",
               app_binary.c_str(), fname);
    else
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass -all %s > %s",
               app_binary.c_str(), fname);
    bool parse_output = true;
    result = yasai_system(command);
    if (result) {
      if (context->get_device()
              ->get_gpgpu()
              ->get_config()
              .experimental_lib_support() &&
          (result == 65280)) {
        // Some CUDA application may exclusively use kernels provided by CUDA
        // libraries (e.g. CUBLAS).  Skipping cuobjdump extraction from the
        // executable for this case.
        // 65280 is the return code from cuobjdump denoting the specific error
        // (tested on CUDA 4.0/4.1/4.2)
        printf("WARNING: Failed to execute: %s\n", command);
        printf("         Executable binary does not contain any GPU kernel.\n");
        parse_output = false;
      } else {
        printf("ERROR: Failed to execute: %s\n", command);
        exit(1);
      }
    }

    if (parse_output) {
      printf("Parsing file %s\n", fname);
      FILE *cuobjdump_in;
      cuobjdump_in = fopen(fname, "r");

      struct cuobjdump_parser parser;
      parser.elfserial = 1;
      parser.ptxserial = 1;
      cuobjdump_lex_init(&(parser.scanner));
      cuobjdump_set_in(cuobjdump_in, (parser.scanner));
      cuobjdump_parse(parser.scanner, &parser, cuobjdumpSectionList);
      cuobjdump_lex_destroy(parser.scanner);
      fclose(cuobjdump_in);
      printf("Done parsing!!!\n");
    } else {
      printf("Parsing skipped for %s\n", fname);
    }

    if (context->get_device()
            ->get_gpgpu()
            ->get_config()
            .experimental_lib_support()) {
      // Experimental library support
      // Currently only for cufft

      std::stringstream cmd;
      cmd << "ldd " << app_binary
          << " | grep $CUDA_INSTALL_PATH | awk \'{print $3}\' > _tempfile_.txt";
      int result = yasai_system(cmd.str().c_str());
      if (result) {
        std::cout << "Failed to execute: " << cmd.str() << std::endl;
        exit(1);
      }
      std::ifstream libsf;
      libsf.open("_tempfile_.txt");
      if (!libsf.is_open()) {
        std::cout << "Failed to open: _tempfile_.txt" << std::endl;
        exit(1);
      }

      // Save the original section list
      std::list<cuobjdumpSection *> tmpsl = cuobjdumpSectionList;
      cuobjdumpSectionList.clear();

      std::string line;
      std::getline(libsf, line);
      std::cout << "DOING: " << line << std::endl;
      int cnt = 1;
      while (libsf.good()) {
        std::stringstream libcodfn;
        libcodfn << "_cuobjdump_complete_lib_" << cnt << "_";
        cmd.str("");  // resetting
        cmd << "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass ";
        cmd << line;
        cmd << " > ";
        cmd << libcodfn.str();
        std::cout << "Running cuobjdump on " << line << std::endl;
        std::cout << "Using command: " << cmd.str() << std::endl;
        result = yasai_system(cmd.str().c_str());
        if (result) {
          printf("ERROR: Failed to execute: %s\n", command);
          exit(1);
        }
        std::cout << "Done" << std::endl;

        std::cout << "Trying to parse " << libcodfn.str() << std::endl;
        FILE *cuobjdump_in;
        cuobjdump_in = fopen(libcodfn.str().c_str(), "r");
        struct cuobjdump_parser parser;
        parser.elfserial = 1;
        parser.ptxserial = 1;
        cuobjdump_lex_init(&(parser.scanner));
        cuobjdump_set_in(cuobjdump_in, (parser.scanner));
        cuobjdump_parse(parser.scanner, &parser, cuobjdumpSectionList);
        cuobjdump_lex_destroy(parser.scanner);
        fclose(cuobjdump_in);
        std::getline(libsf, line);
      }
      libSectionList = cuobjdumpSectionList;

      // Restore the original section list
      cuobjdumpSectionList = tmpsl;
    }
  } else {
    printf(
        "GPGPU-Sim PTX: overriding cuobjdump with '%s' (CUOBJDUMP_SIM_FILE is "
        "set)\n",
        override_cuobjdump);
    snprintf(fname, 1024, "%s", override_cuobjdump);
  }
}

void cuda_runtime_api::extract_code_using_cuobjdump(const char *fn) {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  std::string app_binary = get_app_binary(fn);
  auto ctx_extract_ptx_func = [=](CUctx_st *context) {
    extract_ptx_files_using_cuobjdump(context, fn);
  };
  extract_code_using_cuobjdump_internal(context, app_binary,
                                        ctx_extract_ptx_func);
}

void cuda_runtime_api::extract_code_using_cuobjdump() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  std::string app_binary = get_app_binary();
  auto ctx_extract_ptx_func = [=](CUctx_st *context) {
    extract_ptx_files_using_cuobjdump(context);
  };
  extract_code_using_cuobjdump_internal(context, app_binary,
                                        ctx_extract_ptx_func);
}

//! Read file into char*
// TODO: convert this to C++ streams, will be way cleaner
char *readfile(const std::string filename) {
  assert(filename != "");
  FILE *fp = fopen(filename.c_str(), "r");
  if (!fp) {
    std::cout << "ERROR: Could not open file %s for reading\n"
              << filename << std::endl;
    assert(0);
  }
  // finding size of the file
  int filesize = 0;
  fseek(fp, 0, SEEK_END);

  filesize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  // allocate and copy the entire ptx
  char *ret = (char *)malloc((filesize + 1) * sizeof(char));
  int num = fread(ret, 1, filesize, fp);
  if (num == 0) {
    std::cout << "ERROR: Could not read data from file %s\n"
              << filename << std::endl;
    assert(0);
  }
  ret[filesize] = '\0';
  fclose(fp);
  return ret;
}

//! Function that helps debugging
void printSectionList(std::list<cuobjdumpSection *> sl) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sl.begin(); iter != sl.end(); iter++) {
    (*iter)->print();
  }
}

//! Remove unecessary sm versions from the section list
std::list<cuobjdumpSection *> cuda_runtime_api::pruneSectionList(
    CUctx_st *context) {
  unsigned forced_max_capability = context->get_device()
                                       ->get_gpgpu()
                                       ->get_config()
                                       .get_forced_max_capability();

  // For ptxplus, force the max capability to 19 if it's higher or
  // unspecified(0)
  if (context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus()) {
    if ((forced_max_capability == 0) || (forced_max_capability >= 20)) {
      printf(
          "GPGPU-Sim: WARNING: Capability >= 20 are not supported in "
          "PTXPlus\n\tSetting forced_max_capability to 19\n");
      forced_max_capability = 19;
    }
  }

  std::list<cuobjdumpSection *> prunedList;

  // Find the highest capability (that is lower than the forced maximum) for
  // each cubin file and set it in cuobjdumpSectionMap. Do this only for ptx
  // sections
  std::map<std::string, unsigned> cuobjdumpSectionMap;
  unsigned min_ptx_capability_found = 0;
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (dynamic_cast<cuobjdumpPTXSection *>(*iter) != NULL) {
      if (capability < min_ptx_capability_found ||
          min_ptx_capability_found == 0)
        min_ptx_capability_found = capability;
      if (capability <= forced_max_capability || forced_max_capability == 0) {
        if ((cuobjdumpSectionMap.find((*iter)->getIdentifier()) ==
             cuobjdumpSectionMap.end()) ||
            (cuobjdumpSectionMap[(*iter)->getIdentifier()] < capability))
          cuobjdumpSectionMap[(*iter)->getIdentifier()] = capability;
      }
    }
  }

  // Throw away the sections with the lower capabilites and push those with the
  // highest in the pruned list
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (capability == cuobjdumpSectionMap[(*iter)->getIdentifier()]) {
      prunedList.push_back(*iter);
    } else {
      delete *iter;
    }
  }
  if (prunedList.empty()) {
    printf(
        "Error: No PTX sections found with sm capability that is lower than "
        "current forced maximum capability \n minimum ptx capability found = "
        "%u, maximum forced ptx capability = %u \n User might want to change "
        "either the forced maximum capability from gpgpusim configuration or "
        "update the compilation to generate the required PTX version\n",
        min_ptx_capability_found, forced_max_capability);
    abort();
  }
  return prunedList;
}

//! Merge all PTX sections that have a specific identifier into one file
std::list<cuobjdumpSection *> cuda_runtime_api::mergeMatchingSections(
    std::string identifier) {
  const char *ptxcode = "";
  std::list<cuobjdumpSection *>::iterator old_iter;
  cuobjdumpPTXSection *old_ptxsection = NULL;
  cuobjdumpPTXSection *ptxsection;
  std::list<cuobjdumpSection *> mergedList;

  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL &&
        strcmp(ptxsection->getIdentifier().c_str(), identifier.c_str()) == 0) {
      // Read and remove the last PTX section
      if (old_ptxsection != NULL) {
        ptxcode = readfile(old_ptxsection->getPTXfilename());
        // remove ptx file?
        delete *old_iter;
      }

      // Append all the PTX from the last PTX section into the current PTX
      // section Add 50 to ptxcode to ignore the information regarding
      // version/target/address_size
      if (strlen(ptxcode) >= 50) {
        FILE *ptxfile = fopen((ptxsection->getPTXfilename()).c_str(), "a");
        fprintf(ptxfile, "%s", ptxcode + 50);
        fclose(ptxfile);
      }

      old_iter = iter;
      old_ptxsection = ptxsection;
    }
    // Store all non-PTX sections and PTX sections with non-matching identifiers
    else {
      mergedList.push_back(*iter);
    }
  }

  // Store the final PTX section
  mergedList.push_back(*old_iter);

  return mergedList;
}

//! Merge any PTX sections with matching identifiers
std::list<cuobjdumpSection *> cuda_runtime_api::mergeSections() {
  std::vector<std::string> identifier;
  cuobjdumpPTXSection *ptxsection;

  // Add all identifiers present in PTX sections to a vector
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL) {
      std::string current_id = ptxsection->getIdentifier();

      // If we haven't yet seen a given identifier, add it to the vector
      if (std::find(identifier.begin(), identifier.end(), current_id) ==
          identifier.end()) {
        identifier.push_back(current_id);
      }
    }
  }

  // Call mergeMatchingSections on all identifiers in the vector
  for (std::vector<std::string>::iterator iter = identifier.begin();
       iter != identifier.end(); iter++) {
    cuobjdumpSectionList = mergeMatchingSections(*iter);
  }

  return cuobjdumpSectionList;
}

//! Within the section list, find the ELF section corresponding to a given
//! identifier
cuobjdumpELFSection *findELFSectionInList(
    std::list<cuobjdumpSection *> sectionlist, const std::string identifier) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sectionlist.begin(); iter != sectionlist.end(); iter++) {
    cuobjdumpELFSection *elfsection;
    if ((elfsection = dynamic_cast<cuobjdumpELFSection *>(*iter)) != NULL) {
      if (elfsection->getIdentifier() == identifier) return elfsection;
    }
  }
  return NULL;
}

//! Find an ELF section in all the known lists
cuobjdumpELFSection *cuda_runtime_api::findELFSection(
    const std::string identifier) {
  cuobjdumpELFSection *sec =
      findELFSectionInList(cuobjdumpSectionList, identifier);
  if (sec != NULL) return sec;
  sec = findELFSectionInList(libSectionList, identifier);
  if (sec != NULL) return sec;
  std::cout << "Could not find " << identifier << std::endl;
  assert(0 && "Could not find the required ELF section");
  return NULL;
}

//! Within the section list, find the PTX section corresponding to a given
//! identifier
cuobjdumpPTXSection *cuda_runtime_api::findPTXSectionInList(
    std::list<cuobjdumpSection *> &sectionlist, const std::string identifier) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sectionlist.begin(); iter != sectionlist.end(); iter++) {
    cuobjdumpPTXSection *ptxsection;
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL) {
      if (ptxsection->getIdentifier() == identifier)
        return ptxsection;
      else {
        if (gpgpu_ctx->device_runtime->g_cdp_enabled) {
          printf(
              "Warning: __cudaRegisterFatBinary needs %s, but find PTX section "
              "with %s\n",
              identifier.c_str(), ptxsection->getIdentifier().c_str());
          return ptxsection;
        }
      }
    }
  }
  return NULL;
}

//! Find an PTX section in all the known lists
cuobjdumpPTXSection *cuda_runtime_api::findPTXSection(
    const std::string identifier) {
  cuobjdumpPTXSection *sec =
      findPTXSectionInList(cuobjdumpSectionList, identifier);
  if (sec != NULL) return sec;
  sec = findPTXSectionInList(libSectionList, identifier);
  if (sec != NULL) return sec;
  std::cout << "Could not find " << identifier << std::endl;
  assert(0 && "Could not find the required PTX section");
  return NULL;
}

//! Extract the code using cuobjdump and remove unnecessary sections
void cuda_runtime_api::cuobjdumpInit_internal(
    std::function<void()> ctx_extract_code_func) {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  ctx_extract_code_func();  // extract all the output of cuobjdump to
                            // _cuobjdump_*.*
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0) {
    cuobjdumpSectionList = pruneSectionList(context);
    cuobjdumpSectionList = mergeSections();
  }
}

void cuda_runtime_api::cuobjdumpInit(const char *fn) {
  auto ctx_extract_code_func = [=]() { extract_code_using_cuobjdump(fn); };
  cuobjdumpInit_internal(ctx_extract_code_func);
}

void cuda_runtime_api::cuobjdumpInit() {
  auto ctx_extract_code_func = [=]() { extract_code_using_cuobjdump(); };
  cuobjdumpInit_internal(ctx_extract_code_func);
}

//! Either submit PTX for simulation or convert SASS to PTXPlus and submit it

// ---- yasai: lazy PTX loading ---------------------------------------------------------------
// Architecture ("sm_XX") of the PTX files that YASAI_PTX_LIB_OVERRIDE puts in place of a
// library's own files (their names carry none).
static std::map<std::string, std::string> &yasai_override_arch() {
  static std::map<std::string, std::string> m;
  return m;
}

// A fat binary with many PTX files (libtorch_cuda.so: 418 files, 1.3 GB) used to be parsed in
// full at registration; the parsed form took about 100 bytes per byte of PTX, so a PyTorch
// process passed 58 GB before reaching its first kernel. With lazy loading (YASAI_LAZY_PTX,
// default 1, for fat binaries with more than one selected file) registration only indexes the
// .entry and file-scope .global/.const names of each file; a file is parsed, assembled and its
// ptxas information loaded when one of its kernels is first launched or one of its variables
// is first accessed. PTX files of non-relocatable code are self-contained, so one file holds
// everything its kernels use.
namespace {
// Index of the lazily loaded PTX files of one fat binary.
struct yasai_lazy_ptx_t {
  unsigned handle = 0;                          // fat binary handle the files belong to
  std::vector<std::string> files;               // PTX file paths
  std::vector<std::string> archs;               // "sm_XX" of each file (ptxas target)
  std::vector<bool> loaded;                     // file parsed already (whole-file mode)
  std::unordered_map<std::string, int> owner;   // kernel or variable name -> file index
  // Per-kernel mode (YASAI_PTX_PER_KERNEL, default 1), per file:
  std::vector<bool> vars_loaded;                // file-scope variables and directives parsed
  std::vector<std::set<std::string>> funcs_loaded;  // entries and .func names parsed
  std::vector<int> pieces;                      // number of pieces parsed so far
};
std::vector<yasai_lazy_ptx_t> g_yasai_lazy;     // every lazily registered fat binary

// True when lazy loading is enabled (YASAI_LAZY_PTX unset or nonzero).
bool yasai_lazy_enabled() {
  const char *e = getenv("YASAI_LAZY_PTX");
  return e == NULL || atoi(e) != 0;
}

// Name declared by a file-scope PTX line (".entry", ".global" or ".const" definition), or "".
std::string yasai_ptx_decl_name(const std::string &line) {
  size_t p = 0;
  auto word = [&](const char *w) {
    const size_t n = strlen(w);
    if (line.compare(p, n, w) == 0 && (p + n >= line.size() || isspace((unsigned char)line[p + n]))) {
      p += n;
      while (p < line.size() && isspace((unsigned char)line[p])) ++p;
      return true;
    }
    return false;
  };
  if (line.empty() || line[0] != '.') return "";
  if (word(".extern")) return "";
  word(".visible") || word(".weak");
  const bool entry = word(".entry");
  if (!entry && !(word(".global") || word(".const"))) return "";
  if (entry) {
    size_t e = p;
    while (e < line.size() && (isalnum((unsigned char)line[e]) || line[e] == '_' || line[e] == '$')) ++e;
    return line.substr(p, e - p);
  }
  // variable: the last identifier before '[', '=' or ';'
  size_t end = line.find_first_of("[=;", p);
  if (end == std::string::npos) end = line.size();
  size_t e = end;
  while (e > p && isspace((unsigned char)line[e - 1])) --e;
  size_t b = e;
  while (b > p && (isalnum((unsigned char)line[b - 1]) || line[b - 1] == '_' || line[b - 1] == '$')) --b;
  return line.substr(b, e - b);
}

// Parses file i of L and makes its kernels and variables available.
void yasai_lazy_load_file(gpgpu_context *ctx, yasai_lazy_ptx_t &L, int i) {
  if (L.loaded[i]) return;
  L.loaded[i] = true;
  CUctx_st *context = GPGPUSim_Context(ctx);
  printf("GPGPU-Sim PTX: lazily parsing %s\n", L.files[i].c_str());
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(L.files[i].c_str());
  context->add_binary(symtab, L.handle);
  ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, context->get_device()->get_gpgpu());
  ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT, context->get_device()->get_gpgpu());
  ctx->gpgpu_ptx_info_load_from_filename(L.files[i].c_str(), L.archs[i].c_str());
}

// True when a launch parses only the kernel it needs (YASAI_PTX_PER_KERNEL unset or nonzero)
// rather than the whole PTX file that holds it. PTX reordering (-gpgpu_ptx_reorder, a
// timing-model feature) turns it off: the SASS-guided reorder runs ptxas and cuobjdump on the
// text it parses and maps SASS to PTX line numbers of a whole module, and an external guide
// (GPGPUSIM_SASS_PTXLINE_GUIDE) is keyed to the original file's lines, so a piece of a file
// cannot be guided; those runs keep whole-file lazy parsing.
//   ctx: simulator context (its reorder options come from gpgpusim.config)
bool yasai_per_kernel_enabled(gpgpu_context *ctx) {
  const char *e = getenv("YASAI_PTX_PER_KERNEL");
  if (e != NULL && atoi(e) == 0) return false;
  if (ctx->ptx_reorder_enabled) {
    static bool told = false;
    if (!told) {
      told = true;
      printf("GPGPU-Sim PTX: PTX reordering is on, so PTX files are parsed whole rather than per "
             "kernel\n");
    }
    return false;
  }
  return true;
}

// Kind of a top-level item of a PTX file (see yasai_ptx_split).
enum yasai_ptx_item_kind_t {
  YASAI_PTX_HEADER,     // .version, .target, .address_size or .file line: in every piece
  YASAI_PTX_ENTRY,      // .entry definition with a body
  YASAI_PTX_FUNC_DEF,   // .func definition with a body
  YASAI_PTX_FUNC_DECL,  // .func prototype, including .extern .func
  YASAI_PTX_DEBUG,      // .section of debug information: never parsed
  YASAI_PTX_OTHER       // file-scope variable or other directive: in the first piece only
};

// One top-level item of a PTX file.
struct yasai_ptx_item_t {
  size_t begin = 0;                         // first byte of the item in the file text
  size_t end = 0;                           // one past the last byte
  yasai_ptx_item_kind_t kind = YASAI_PTX_OTHER;  // what the item is
  std::string name;                         // entry or function name; empty otherwise
};

// Position of the first character at or after p that is neither white space nor inside a
// comment.
//   t: PTX text; p: start position
size_t yasai_ptx_skip_space(const std::string &t, size_t p) {
  const size_t n = t.size();
  while (p < n) {
    if (isspace((unsigned char)t[p])) {
      ++p;
    } else if (t.compare(p, 2, "//") == 0) {
      p = t.find('\n', p);
      if (p == std::string::npos) p = n;
    } else if (t.compare(p, 2, "/*") == 0) {
      p = t.find("*/", p + 2);
      p = p == std::string::npos ? n : p + 2;
    } else {
      break;
    }
  }
  return p;
}

// True when c can be part of a PTX identifier.
//   c: character
bool yasai_ptx_ident_char(char c) {
  return isalnum((unsigned char)c) || c == '_' || c == '$';
}

// Identifier that starts at p (after white space), or "".
//   t: PTX text; p: position to read from (advanced past the identifier)
std::string yasai_ptx_read_ident(const std::string &t, size_t &p) {
  p = yasai_ptx_skip_space(t, p);
  const size_t b = p;
  while (p < t.size() && yasai_ptx_ident_char(t[p])) ++p;
  return t.substr(b, p - b);
}

// Splits PTX text into its top-level items, in file order. An item ends at a semicolon at
// brace depth 0, or at the closing brace of a body that no semicolon follows (a function body;
// a variable initializer "= { ... };" ends at its semicolon). Header directives, which carry no
// semicolon, end at the end of their line.
//   t: PTX text of one file
std::vector<yasai_ptx_item_t> yasai_ptx_split(const std::string &t) {
  std::vector<yasai_ptx_item_t> items;
  const size_t n = t.size();
  size_t i = 0;
  while (true) {
    i = yasai_ptx_skip_space(t, i);
    if (i >= n) break;
    yasai_ptx_item_t it;
    it.begin = i;
    if (t.compare(i, 8, ".version") == 0 || t.compare(i, 7, ".target") == 0 ||
        t.compare(i, 13, ".address_size") == 0 || t.compare(i, 5, ".file") == 0) {
      size_t e = t.find('\n', i);
      if (e == std::string::npos) e = n;
      it.end = e;
      it.kind = YASAI_PTX_HEADER;
      items.push_back(it);
      i = e;
      continue;
    }
    int depth = 0;
    bool body = false;
    size_t p = i;
    while (p < n) {
      const char c = t[p];
      if (c == '/' && (t.compare(p, 2, "//") == 0 || t.compare(p, 2, "/*") == 0)) {
        const size_t q = yasai_ptx_skip_space(t, p);
        p = q > p ? q : p + 1;
      } else if (c == '"') {
        const size_t q = t.find('"', p + 1);
        p = q == std::string::npos ? n : q + 1;
      } else if (c == '{') {
        ++depth;
        body = true;
        ++p;
      } else if (c == '}') {
        --depth;
        ++p;
        if (depth == 0) {
          size_t q = p;
          while (q < n && isspace((unsigned char)t[q])) ++q;
          if (q < n && t[q] == ';') continue;  // initializer: the semicolon ends it
          break;
        }
      } else if (c == ';' && depth == 0) {
        ++p;
        break;
      } else {
        ++p;
      }
    }
    it.end = p;
    // classify: skip linkage words, then look at the declaring directive
    size_t q = it.begin;
    std::string word;
    while (true) {
      q = yasai_ptx_skip_space(t, q);
      size_t e = q;
      if (e < n && t[e] == '.') ++e;
      while (e < it.end && yasai_ptx_ident_char(t[e])) ++e;
      word = t.substr(q, e - q);
      if (word == ".visible" || word == ".weak" || word == ".extern" || word == ".common") {
        q = e;
        continue;
      }
      q = e;
      break;
    }
    if (word == ".entry") {
      it.name = yasai_ptx_read_ident(t, q);
      it.kind = body ? YASAI_PTX_ENTRY : YASAI_PTX_OTHER;
    } else if (word == ".func") {
      q = yasai_ptx_skip_space(t, q);
      if (q < n && t[q] == '(') {  // return parameter list
        int d = 0;
        for (; q < it.end; ++q) {
          if (t[q] == '(') ++d;
          if (t[q] == ')' && --d == 0) {
            ++q;
            break;
          }
        }
      }
      it.name = yasai_ptx_read_ident(t, q);
      it.kind = body ? YASAI_PTX_FUNC_DEF : YASAI_PTX_FUNC_DECL;
    } else if (word == ".section") {
      size_t r = yasai_ptx_skip_space(t, q);
      it.kind = t.compare(r, 6, ".debug") == 0 ? YASAI_PTX_DEBUG : YASAI_PTX_OTHER;
    }
    items.push_back(it);
    i = p;
  }
  return items;
}

// Adds to out every identifier of t[b, e) that names a function of the file. Identifiers that
// follow '%' (registers) or '.' (directive and opcode parts) are skipped.
//   t: PTX text; b, e: byte range; funcs: function names of the file; out: names found
void yasai_ptx_called(const std::string &t, size_t b, size_t e,
                      const std::unordered_map<std::string, std::vector<int>> &funcs,
                      std::vector<std::string> &out) {
  size_t p = b;
  while (p < e) {
    if (!yasai_ptx_ident_char(t[p]) || isdigit((unsigned char)t[p])) {
      ++p;
      continue;
    }
    const char before = p > b ? t[p - 1] : ' ';
    const size_t s = p;
    while (p < e && yasai_ptx_ident_char(t[p])) ++p;
    if (before == '%' || before == '.') continue;
    std::string id = t.substr(s, p - s);
    if (funcs.count(id)) out.push_back(std::move(id));
  }
}

// Total bytes of PTX parsed by the per-kernel loader (printed with each piece).
size_t g_yasai_piece_bytes = 0;

// Per-kernel loading: parses from file i of L only what name needs. For a kernel that is the
// .entry, the .func definitions it reaches (directly, or through the function pointers in the
// file's variable initializers), and, in the file's first piece, the file-scope variables and
// directives. Later pieces of the same file parse into the same global symbol table, so they
// share the file's variables (one copy per file, as with whole-file parsing) and the functions
// already parsed. For a variable, the first piece without an entry. ptxas runs on a
// self-contained copy of the piece (all variables, every reached function, the entry), whose
// resource usage is the kernel's. Returns true when something was parsed.
//   ctx: simulator context; L: index of the fat binary; i: file index; name: kernel or variable
bool yasai_lazy_load_piece(gpgpu_context *ctx, yasai_lazy_ptx_t &L, int i, const char *name) {
  std::string target = name;
  if (L.funcs_loaded[i].count(target)) return false;
  std::ifstream in(L.files[i]);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string t = ss.str();
  const std::vector<yasai_ptx_item_t> items = yasai_ptx_split(t);
  std::unordered_map<std::string, std::vector<int>> funcs;  // function name -> its items
  int entry_item = -1;
  for (int k = 0; k < (int)items.size(); ++k) {
    if (items[k].kind == YASAI_PTX_FUNC_DEF || items[k].kind == YASAI_PTX_FUNC_DECL)
      funcs[items[k].name].push_back(k);
    if (items[k].kind == YASAI_PTX_ENTRY && items[k].name == target) entry_item = k;
  }
  if (entry_item < 0 && L.vars_loaded[i]) return false;  // a variable, already parsed
  // functions reached from the entry and from the variable initializers
  std::vector<std::string> work;
  if (entry_item >= 0) yasai_ptx_called(t, items[entry_item].begin, items[entry_item].end, funcs, work);
  for (const yasai_ptx_item_t &it : items)
    if (it.kind == YASAI_PTX_OTHER) yasai_ptx_called(t, it.begin, it.end, funcs, work);
  std::set<std::string> reached;
  while (!work.empty()) {
    std::string f = work.back();
    work.pop_back();
    if (!reached.insert(f).second) continue;
    for (int k : funcs[f])
      if (items[k].kind == YASAI_PTX_FUNC_DEF)
        yasai_ptx_called(t, items[k].begin, items[k].end, funcs, work);
  }
  std::string parse_text, ptxas_text;
  for (int k = 0; k < (int)items.size(); ++k) {
    const yasai_ptx_item_t &it = items[k];
    bool to_ptxas = false, to_parse = false;
    switch (it.kind) {
      case YASAI_PTX_HEADER:
        to_ptxas = to_parse = true;
        break;
      case YASAI_PTX_DEBUG:
        break;
      case YASAI_PTX_OTHER:
        to_ptxas = true;
        to_parse = !L.vars_loaded[i];
        break;
      case YASAI_PTX_FUNC_DEF:
      case YASAI_PTX_FUNC_DECL:
        to_ptxas = reached.count(it.name) > 0;
        to_parse = to_ptxas && !L.funcs_loaded[i].count(it.name);
        break;
      case YASAI_PTX_ENTRY:
        to_ptxas = to_parse = (k == entry_item);
        break;
    }
    if (to_ptxas) ptxas_text.append(t, it.begin, it.end - it.begin).push_back('\n');
    if (to_parse) parse_text.append(t, it.begin, it.end - it.begin).push_back('\n');
  }
  const int piece = L.pieces[i]++;
  const std::string parse_file = L.files[i] + ".k" + std::to_string(piece) + ".ptx";
  {
    std::ofstream out(parse_file);
    out << parse_text;
  }
  g_yasai_piece_bytes += parse_text.size();
  printf("GPGPU-Sim PTX: parsing %s of %s (piece %d: %zu of %zu bytes; %zu bytes parsed in "
         "pieces so far)\n",
         entry_item >= 0 ? target.c_str() : "the variables", L.files[i].c_str(), piece,
         parse_text.size(), t.size(), g_yasai_piece_bytes);
  CUctx_st *context = GPGPUSim_Context(ctx);
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(parse_file.c_str());
  context->add_binary(symtab, L.handle);
  ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, context->get_device()->get_gpgpu());
  ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT, context->get_device()->get_gpgpu());
  L.vars_loaded[i] = true;
  for (const std::string &f : reached) L.funcs_loaded[i].insert(f);
  if (entry_item >= 0) {
    L.funcs_loaded[i].insert(target);
    const std::string ptxas_file = L.files[i] + ".k" + std::to_string(piece) + ".full.ptx";
    {
      std::ofstream out(ptxas_file);
      out << ptxas_text;
    }
    ctx->gpgpu_ptx_info_load_from_filename(ptxas_file.c_str(), L.archs[i].c_str());
  }
  return true;
}

// yasai_func::g_lazy_load_name: loads the file that defines name (whole-file mode) or the part
// of it that name needs (per-kernel mode).
bool yasai_lazy_load_name(const char *name) {
  for (yasai_lazy_ptx_t &L : g_yasai_lazy) {
    auto it = L.owner.find(name);
    if (it == L.owner.end()) continue;
    if (yasai_per_kernel_enabled(GPGPU_Context())) {
      if (L.loaded[it->second]) return false;
      return yasai_lazy_load_piece(GPGPU_Context(), L, it->second, name);
    }
    if (!L.loaded[it->second]) {
      yasai_lazy_load_file(GPGPU_Context(), L, it->second);
      return true;
    }
  }
  return false;
}
}  // namespace

// yasai: a library that registers fat binaries but is not named in PYTORCH_BIN (a module that
// FlashInfer or sglang compiled at run time and loaded with dlopen, or a framework extension)
// used to have no PTX: the PTX of PYTORCH_BIN is extracted once, at the first registration.
// Now the library file of every registered fat binary is found with dladdr; the first time a
// library is seen, the PTX files that cuobjdump extracts from it (one architecture, selected as
// for PYTORCH_BIN) are indexed for lazy loading like the PYTORCH_BIN files, so its kernels are
// parsed at their first launch. YASAI_DYNAMIC_LIB_PTX=0 turns this off.
//   ctx       simulator context
//   handle    fat binary handle of this registration
//   fatCubin  the fat binary wrapper the library passed (its address locates the library)
static void yasai_register_library_ptx(gpgpu_context *ctx, unsigned handle, void *fatCubin) {
  const char *off = getenv("YASAI_DYNAMIC_LIB_PTX");
  if (fatCubin == NULL || !yasai_lazy_enabled() || (off != NULL && atoi(off) == 0)) return;
  Dl_info info;
  if (!dladdr(fatCubin, &info) || info.dli_fname == NULL || info.dli_fname[0] == '\0') return;
  char resolved[4096];
  const std::string path = realpath(info.dli_fname, resolved) ? resolved : info.dli_fname;
  static std::set<std::string> seen;
  static std::set<std::string> listed;
  static bool listed_init = false;
  if (!listed_init) {
    listed_init = true;
    if (const char *pb = getenv("PYTORCH_BIN")) {
      std::string list(pb);
      size_t start = 0;
      while (start <= list.size()) {
        const size_t end = std::min(list.find(':', start), list.size());
        if (end > start) {
          const std::string b = list.substr(start, end - start);
          listed.insert(realpath(b.c_str(), resolved) ? std::string(resolved) : b);
        }
        start = end + 1;
      }
    }
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
      exe[n] = '\0';
      listed.insert(realpath(exe, resolved) ? std::string(resolved) : std::string(exe));
    }
  }
  if (listed.count(path) || !seen.insert(path).second) return;
  const unsigned cap =
      GPGPUSim_Context(ctx)->get_device()->get_gpgpu()->get_config().get_forced_max_capability();
  const std::vector<std::string> files = yasai_jit::ptx_files_of_library(path, cap);
  printf("GPGPU-Sim PTX: library %s registered after start-up: %zu PTX files indexed\n",
         path.c_str(), files.size());
  if (files.empty()) return;
  yasai_lazy_ptx_t L;
  L.handle = handle;
  for (const std::string &f : files) {
    const int idx = int(L.files.size());
    L.files.push_back(f);
    L.archs.push_back(yasai_jit::arch_of_ptx_file(f));
    L.loaded.push_back(false);
    // per-kernel parsing state, one entry per file as in cuobjdumpParseBinary
    L.vars_loaded.push_back(false);
    L.funcs_loaded.emplace_back();
    L.pieces.push_back(0);
    std::ifstream in(f);
    std::string line;
    while (std::getline(in, line)) {
      const std::string nm = yasai_ptx_decl_name(line);
      if (!nm.empty()) L.owner.emplace(nm, idx);
    }
  }
  g_yasai_lazy.push_back(std::move(L));
  yasai_func::g_lazy_load_name = yasai_lazy_load_name;
}

void gpgpu_context::cuobjdumpParseBinary(unsigned int handle) {
  CUctx_st *context = GPGPUSim_Context(this);
  if (api->fatbin_registered[handle]) return;
  api->fatbin_registered[handle] = true;
  std::string fname = api->fatbinmap[handle];

  if (api->name_symtab.find(fname) != api->name_symtab.end()) {
    symbol_table *symtab = api->name_symtab[fname];
    context->add_binary(symtab, handle);
    return;
  }
  symbol_table *symtab = NULL;

#if (CUDART_VERSION >= 6000)
  // Build selected_files based on forced_max_capability and suffix rules
  std::vector<std::string> selected_files;
  unsigned forced_max_capability = context->get_device()
                                       ->get_gpgpu()
                                       ->get_config()
                                       .get_forced_max_capability();
  // yasai: the version is selected per library (file name prefix before ".<n>.sm_"), so the PTX
  // of libraries built for different architectures (PYTORCH_BIN with several libraries) all load.
  std::map<std::string, std::map<unsigned, std::set<std::string>>> by_library;
  for (const auto &kv : api->version_filename)
    for (const std::string &f : kv.second) {
      const size_t sm_pos = f.find("sm_");
      const size_t cut = (sm_pos != std::string::npos && sm_pos >= 2) ? f.rfind('.', sm_pos - 2)
                                                                       : std::string::npos;
      by_library[cut == std::string::npos ? std::string() : f.substr(0, cut)][kv.first].insert(f);
    }
  // yasai: YASAI_PTX_LIB_OVERRIDE="<library prefix>=<PTX file>[,...]" replaces the selected PTX
  // of one library by one file (for example NCCL's rdc kernels merged into one module), while the
  // other libraries keep theirs; the replacement keeps the architecture of the first file it
  // replaces (for ptxas).
  std::map<std::string, std::string> lib_override;
  if (const char *ov = getenv("YASAI_PTX_LIB_OVERRIDE")) {
    std::string list(ov);
    size_t start = 0;
    while (start < list.size()) {
      size_t end = list.find(',', start);
      if (end == std::string::npos) end = list.size();
      const std::string item = list.substr(start, end - start);
      const size_t eq = item.find('=');
      if (eq != std::string::npos) lib_override[item.substr(0, eq)] = item.substr(eq + 1);
      start = end + 1;
    }
  }
  for (auto &library : by_library) {
  std::map<unsigned, std::set<std::string>> &vf = library.second;
  const size_t first_of_library = selected_files.size();
  if (!vf.empty()) {
    // pick selected_version: largest version <= forced_max_capability (or
    // largest available)
    unsigned selected_version = 0;
    if (forced_max_capability != 0) {
      for (auto rit = vf.rbegin();
           rit != vf.rend(); ++rit) {
        if (rit->first <= forced_max_capability) {
          selected_version = rit->first;
          break;
        }
      }
      if (selected_version == 0)
        selected_version = vf.rbegin()->first;
    } else {
      selected_version = vf.rbegin()->first;
    }

    // Determine if there are suffix variants for this selected_version
    bool has_suffix = false;
    for (auto &check_name : vf[selected_version]) {
      size_t sm_pos = check_name.find("sm_");
      if (sm_pos != std::string::npos) {
        size_t dot_pos = check_name.find('.', sm_pos);
        std::string arch_part =
            check_name.substr(sm_pos + 3, dot_pos - sm_pos - 3);
        for (size_t j = 0; j < arch_part.length(); j++) {
          if (!std::isdigit(static_cast<unsigned char>(arch_part[j]))) {
            has_suffix = true;
            break;
          }
        }
        if (has_suffix) break;
      }
    }

    for (auto &ptx_filename : vf[selected_version]) {
      // Skip base version if suffix version exists
      if (has_suffix) {
        size_t sm_pos = ptx_filename.find("sm_");
        if (sm_pos != std::string::npos) {
          size_t dot_pos = ptx_filename.find('.', sm_pos);
          std::string arch_part =
              ptx_filename.substr(sm_pos + 3, dot_pos - sm_pos - 3);
          bool is_base = true;
          for (size_t j = 0; j < arch_part.length(); j++) {
            if (!std::isdigit(static_cast<unsigned char>(arch_part[j]))) {
              is_base = false;
              break;
            }
          }
          if (is_base) {
            printf("GPGPU-Sim PTX: Skipping %s (suffix version available)\n",
                   ptx_filename.c_str());
            continue;
          }
        }
      }
      selected_files.push_back(ptx_filename);
    }
  }
  for (const auto &ov : lib_override) {
    const std::string &lib = library.first;
    const size_t base = lib.rfind('/') == std::string::npos ? 0 : lib.rfind('/') + 1;
    if (lib.compare(base, ov.first.size(), ov.first) != 0 || selected_files.size() == first_of_library)
      continue;
    const std::string first = selected_files[first_of_library];
    const size_t sm_pos = first.find("sm_");
    yasai_override_arch()[ov.second] =
        sm_pos == std::string::npos ? "" : first.substr(sm_pos, first.find('.', sm_pos) - sm_pos);
    printf("GPGPU-Sim PTX: %zu PTX files of %s replaced by %s\n",
           selected_files.size() - first_of_library, lib.c_str(), ov.second.c_str());
    selected_files.resize(first_of_library);
    selected_files.push_back(ov.second);
    break;
  }

  }
  // yasai: an override for a library that carries no PTX of its own adds its file. A library
  // built with separable compilation (bitsandbytes) keeps the PTX in its objects' relocatable
  // device code and registers only device-linked cubins, so cuobjdump lists no PTX for it and
  // the loop above never sees it. The architecture comes from the file's .target line.
  for (const auto &ov : lib_override) {
    bool listed = false;
    for (const auto &library : by_library) {
      const std::string &lib = library.first;
      const size_t base = lib.rfind('/') == std::string::npos ? 0 : lib.rfind('/') + 1;
      listed = listed || lib.compare(base, ov.first.size(), ov.first) == 0;
    }
    if (listed) continue;
    std::ifstream in(ov.second);
    std::string line, arch;
    while (arch.empty() && std::getline(in, line)) {
      const size_t t = line.find(".target");
      const size_t sm = line.find("sm_", t == std::string::npos ? line.size() : t);
      if (t != std::string::npos && sm != std::string::npos) {
        size_t e = sm + 3;
        while (e < line.size() && isalnum((unsigned char)line[e])) ++e;
        arch = line.substr(sm, e - sm);
      }
    }
    if (arch.empty()) {
      printf("GPGPU-Sim PTX: override %s has no .target line, skipped\n", ov.second.c_str());
      continue;
    }
    yasai_override_arch()[ov.second] = arch;
    selected_files.push_back(ov.second);
    printf("GPGPU-Sim PTX: %s carries no PTX; %s (%s) added for it\n", ov.first.c_str(),
           ov.second.c_str(), arch.c_str());
  }

  // Parse the selected files (if any)
  const char *selected_ptx_override = getenv("GPGPUSIM_SELECTED_PTX_OVERRIDE");
  const bool use_selected_ptx_override =
      selected_ptx_override != NULL && strlen(selected_ptx_override) != 0;
  // yasai: the override file replaces all selected files together (for example the PTX of an
  // rdc library merged into one module), so it is parsed once rather than once per file.
  if (use_selected_ptx_override && selected_files.size() > 1) selected_files.resize(1);
  if (GPTRACE(PTX_IR)) {
    char cwd_buf[4096];
    [[maybe_unused]] const char *cwd =
        getcwd(cwd_buf, sizeof(cwd_buf)) ? cwd_buf : "<getcwd failed>";
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: selected-file path cwd=%s fname=%s handle=%u\n",
                 cwd, fname.c_str(), handle);
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: env GPGPUSIM_SELECTED_PTX_OVERRIDE=%s\n",
                 selected_ptx_override ? selected_ptx_override : "<unset>");
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: env PTX_SIM_USE_PTX_FILE=%s\n",
                 getenv("PTX_SIM_USE_PTX_FILE") ? getenv("PTX_SIM_USE_PTX_FILE")
                                                : "<unset>");
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: env PTX_SIM_KERNELFILE=%s\n",
                 getenv("PTX_SIM_KERNELFILE") ? getenv("PTX_SIM_KERNELFILE")
                                              : "<unset>");
    GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                 "PTX load: selected_files=%zu override_enabled=%d\n",
                 selected_files.size(), use_selected_ptx_override ? 1 : 0);
    for (size_t i = 0; i < selected_files.size(); ++i) {
      GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                   "PTX load: selected_files[%zu]=%s\n", i,
                   selected_files[i].c_str());
    }
  }

  // yasai: lazy loading (see yasai_lazy_ptx_t): index the files and parse none of them now.
  if (!use_selected_ptx_override && selected_files.size() > 1 && yasai_lazy_enabled()) {
    yasai_lazy_ptx_t L;
    L.handle = handle;
    for (auto &f : selected_files) {
      const int idx = int(L.files.size());
      L.files.push_back(f);
      std::string arch = "";
      size_t sm_pos = f.find("sm_");
      if (yasai_override_arch().count(f)) arch = yasai_override_arch()[f];
      else if (sm_pos != std::string::npos) arch = f.substr(sm_pos, f.find('.', sm_pos) - sm_pos);
      L.archs.push_back(arch);
      L.loaded.push_back(false);
      L.vars_loaded.push_back(false);
      L.funcs_loaded.emplace_back();
      L.pieces.push_back(0);
      std::ifstream in(f);
      std::string line;
      while (std::getline(in, line)) {
        const std::string n = yasai_ptx_decl_name(line);
        if (!n.empty()) L.owner.emplace(n, idx);
      }
    }
    printf("GPGPU-Sim PTX: lazy PTX loading: %zu files, %zu kernels and variables indexed\n",
           L.files.size(), L.owner.size());
    g_yasai_lazy.push_back(std::move(L));
    yasai_func::g_lazy_load_name = yasai_lazy_load_name;
    symtab = gpgpu_ptx_sim_load_ptx_from_string(
        ".version 8.0\n.target sm_90\n.address_size 64\n", 900000 + handle);
    api->name_symtab[fname] = symtab;
    context->add_binary(symtab, handle);
    return;
  }

  for (auto &ptx_filename : selected_files) {
    const char *parse_filename =
        use_selected_ptx_override ? selected_ptx_override : ptx_filename.c_str();
    if (use_selected_ptx_override) {
      printf("GPGPU-Sim PTX: overriding selected PTX %s with %s\n",
             ptx_filename.c_str(), selected_ptx_override);
    }
    if (GPTRACE(PTX_IR)) {
      GPPRINTF_GPU(context->get_device()->get_gpgpu(), PTX_IR,
                   "PTX load: parse selected=%s parse_filename=%s\n",
                   ptx_filename.c_str(), parse_filename);
    }
    printf("GPGPU-Sim PTX: Parsing %s\n", parse_filename);
    symtab = gpgpu_ptx_sim_load_ptx_from_filename(parse_filename);
  }
  api->name_symtab[fname] = symtab;
  context->add_binary(symtab, handle);
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  // Load PTXInfo only for the selected files (if any)
  for (auto &ptx_filename : selected_files) {
    // Extract full architecture string (e.g., "sm_120a" from "kernel.2.sm_120a.ptx")
    std::string arch_str = "";
    size_t sm_pos = ptx_filename.find("sm_");
    if (sm_pos != std::string::npos) {
      size_t dot_pos = ptx_filename.find('.', sm_pos);
      arch_str = ptx_filename.substr(sm_pos, dot_pos - sm_pos);
    }
    const char *ptxinfo_filename = use_selected_ptx_override
                                       ? selected_ptx_override
                                       : ptx_filename.c_str();
    if (GPTRACE(PTX_IR)) {
      GPPRINTF_GPU(
          context->get_device()->get_gpgpu(), PTX_IR,
          "PTX load: ptxinfo selected=%s ptxinfo_filename=%s arch=%s\n",
          ptx_filename.c_str(), ptxinfo_filename, arch_str.c_str());
    }
    printf("GPGPU-Sim PTX: Loading PTXInfo from %s\n", ptxinfo_filename);
    gpgpu_ptx_info_load_from_filename(ptxinfo_filename, arch_str.c_str());
  }
  return;
#endif

  unsigned max_capability = 0;
  for (std::list<cuobjdumpSection *>::iterator iter =
           api->cuobjdumpSectionList.begin();
       iter != api->cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (capability > max_capability) max_capability = capability;
  }
  if (max_capability > 20)
    printf("WARNING: No guarantee that PTX will be parsed for SM version %u\n",
           max_capability);
  if (max_capability == 0)
    max_capability = context->get_device()
                         ->get_gpgpu()
                         ->get_config()
                         .get_forced_max_capability();

  cuobjdumpPTXSection *ptx = NULL;
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0)
    ptx = api->findPTXSection(fname);
  char *ptxcode;
  const char *override_ptx_name = getenv("PTX_SIM_KERNELFILE");
  if (override_ptx_name == NULL or getenv("PTX_SIM_USE_PTX_FILE") == NULL or
      strlen(getenv("PTX_SIM_USE_PTX_FILE")) == 0) {
    ptxcode = readfile(ptx->getPTXfilename());
  } else {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with '%s' "
        "(PTX_SIM_USE_PTX_FILE is set)\n",
        override_ptx_name);
    ptxcode = readfile(override_ptx_name);
  }
  if (context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus()) {
    cuobjdumpELFSection *elfsection = api->findELFSection(ptx->getIdentifier());
    assert(elfsection != NULL);
    char *ptxplus_str = ptxinfo->gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(
        ptx->getPTXfilename(), elfsection->getELFfilename(),
        elfsection->getSASSfilename());
    symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxplus_str, handle);
    printf("Adding %s with cubin handle %u\n", ptx->getPTXfilename().c_str(),
           handle);
    context->add_binary(symtab, handle);
    gpgpu_ptxinfo_load_from_string(ptxcode, handle, max_capability,
                                   context->no_of_ptx);
    delete[] ptxplus_str;
  } else {
    symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxcode, handle);
    context->add_binary(symtab, handle);
    gpgpu_ptxinfo_load_from_string(ptxcode, handle, max_capability,
                                   context->no_of_ptx);
  }
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  api->name_symtab[fname] = symtab;

  // TODO: Remove temporarily files as per configurations
}
}

extern "C" {

void **CUDARTAPI __cudaRegisterFatBinarySST(const char *fn) {
  return cudaRegisterFatBinaryInternal(fn, NULL, NULL);
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaRegisterFatBinaryInternal(fatCubin);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st *stream = 0) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
  return 0;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  gpgpu_context *ctx = GPGPU_Context();
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config &config = ctx->api->g_cuda_launch_stack.back();
  if (gridDim) {
    *gridDim = config.grid_dim();
  }
  if (blockDim) {
    *blockDim = config.block_dim();
  }
  if (sharedMem) {
    *sharedMem = config.shared_mem();
  }
  if (stream) {
    *reinterpret_cast<cudaStream_t *>(stream) = config.get_stream();
  }
  return g_last_cudaError = cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunctionSST(unsigned fatCubinHandle,
                                         uint64_t hostFun,
                                         char deviceFun[512]) {
  cudaRegisterFunctionInternal((void **)(uintptr_t)fatCubinHandle,
                               (const char *)(uintptr_t)hostFun,
                               (char *)deviceFun, NULL, 0, NULL, NULL, NULL,
                               NULL);
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim) {
  cudaRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun, deviceName,
                               thread_limit, tid, bid, bDim, gDim);
}

cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t *kernelPtr,
                                      const void *entryFuncAddr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *kernelPtr = reinterpret_cast<cudaKernel_t>(const_cast<void *>(entryFuncAddr));
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                                         dim3 blockDim, void **args,
                                         size_t sharedMem,
                                         cudaStream_t stream) {
  return cudaLaunchKernelInternal(
      reinterpret_cast<const char *>(kernel), gridDim, blockDim,
      const_cast<const void **>(args), sharedMem, stream);
}

cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel,
                                              dim3 gridDim, dim3 blockDim,
                                              void **args, size_t sharedMem,
                                              cudaStream_t stream) {
  return __cudaLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
}

extern void __cudaRegisterVar(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global) {
  cudaRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress, deviceName,
                          ext, size, constant, global);
}

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem,
                                                 cudaStream_t stream) {
  return cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

cudaError_t cudaDeviceReset(void) {
  // Should reset the simulated GPU
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
  if (!yasai_rec::Recorder::get().on()) return cudaDeviceSynchronizeInternal();
  const uint64_t t0 = yasai_rec::mono_ns();
  const uint32_t corr = yasai_rec::Recorder::get().corr();
  const cudaError_t ret = cudaDeviceSynchronizeInternal();
  yasai_rec_sync("cudaDeviceSynchronize", t0, corr, ret, yasai_rec::kContextSync, nullptr, 0, 0);
  return ret;
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterShared\n");
}

void CUDARTAPI __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
                                       size_t size, size_t alignment,
                                       int storage) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n");
}

void __cudaRegisterTexture(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext)  // passes in a newly created textureReference
{
  __cudaRegisterTextureInternal(fatCubinHandle, hostVar, deviceAddress,
                                deviceName, dim, norm, ext);
}

char __cudaInitModule(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufferObj) {
  return cudaGLMapBufferObjectInternal(devPtr, bufferObj);
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj) {
  return cudaGLUnmapBufferObjectInternal(bufferObj);
}

cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 2010)

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t bytes,
                                    unsigned int flags) {
  return cudaHostAllocInternal(pHost, bytes, flags);
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                               unsigned int flags) {
  return cudaHostGetDevicePointerInternal(pDevice, pHost, flags);
}

// yasai: managed memory is mapped host memory, which kernels access in place (device address ==
// host address), so the host and the device see one copy as with unified memory. cudaFree
// releases it. bitsandbytes (paged optimizers) uses it.
//   devPtr: receives the allocation; size: bytes; flags: cudaMemAttachGlobal or
//   cudaMemAttachHost (both give the same mapping here)
__host__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)flags;
  if (devPtr == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  const cudaError_t e = cudaHostAllocInternal(devPtr, size, cudaHostAllocMapped);
  if (e != cudaSuccess) return e;
  std::lock_guard<std::mutex> lock(g_yasai_managed_mutex);
  g_yasai_managed.insert(*devPtr);
  return g_last_cudaError = cudaSuccess;
}

// yasai: prefetching moves managed pages between host and device on a real GPU. Here managed
// memory is one host copy that kernels access in place, so there is nothing to move; the call
// only checks its arguments.
//   devPtr: start of the range; count: bytes; location: destination; flags: must be 0;
//   stream: stream the prefetch would be ordered on
__host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                                    struct cudaMemLocation location,
                                                    unsigned int flags, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)count;
  (void)location;
  (void)stream;
  if (devPtr == NULL || flags != 0) return g_last_cudaError = cudaErrorInvalidValue;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: minimal answer from the address ranges the simulator knows: host_window ranges are
  // registered host memory, addresses at or above GLOBAL_HEAP_START are device memory, everything
  // else is unregistered host memory.
  if (attributes == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  const uint64_t a = reinterpret_cast<uint64_t>(ptr);
  memset(attributes, 0, sizeof(*attributes));
  if (yasai_vmm::is_pinned_host(a)) {
    attributes->type = cudaMemoryTypeHost;
    attributes->device = 0;
    attributes->devicePointer = const_cast<void *>(ptr);
    attributes->hostPointer = const_cast<void *>(ptr);
  } else if ((a >= GLOBAL_HEAP_START && a < 0x500000000000ull) || host_window::contains(a, 1)) {
    attributes->type = cudaMemoryTypeDevice;
    attributes->device = 0;
    attributes->devicePointer = const_cast<void *>(ptr);
    attributes->hostPointer = NULL;
  } else {
    attributes->type = cudaMemoryTypeUnregistered;
    attributes->device = -2;
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer,
                                                       int device,
                                                       int peerDevice) {
  // yasai: one simulated device per process, so no peer access.
  (void)device;
  (void)peerDevice;
  if (canAccessPeer) *canAccessPeer = 0;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice,
                                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: memory shared between rank processes needs no peer enablement.
  (void)peerDevice;
  (void)flags;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags(int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // This flag is implicitly always on (unless you are using the driver API). It
  // is safe for GPGPU-Sim to just ignore it.
  if (cudaDeviceMapHost == flags) {
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                            const char *hostFun) {
  return cudaFuncGetAttributesInternal(attr, hostFun);
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
  CUevent_st *e = new CUevent_st((flags & cudaEventBlockingSync) != 0);
  // yasai: an interprocess event keeps its counters in shared memory (cudaIpcGetEventHandle).
  if (flags & cudaEventInterprocess) {
    unsigned int *counters = nullptr;
    char ref[64];
    if (!yasai_vmm::ipc_event_create(&counters, ref)) {
      delete e;
      return g_last_cudaError = cudaErrorMemoryAllocation;
    }
    e->set_shared(counters, ref);
  }
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *driverVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *runtimeVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

#if CUDART_VERSION >= 3000
__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) {
  return cudaFuncSetCacheConfigInternal(func, cacheConfig);
}

// Jin: hack for cdp
__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit,
                                                  size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

// #if CUDART_VERSION >= 9000
//__host__  cudaError_t cudaFuncSetAttribute ( const void* func, enum
//  cudaFuncAttribute attr, int value ) {

// ignore this Attribute for now, and the default is that carveout =
// cudaSharedmemCarveoutDefault;   //  (-1)
//	return g_last_cudaError = cudaSuccess;
//}

#endif

#endif

static bool max_dynamic_shared_memory_fits(
    CUctx_st *context, function_info *entry, unsigned requested_dynamic,
    unsigned *static_smem, unsigned *max_per_block_optin) {
  const struct gpgpu_ptx_sim_info *kinfo = entry->get_kernel_info();
  *static_smem = kinfo ? kinfo->smem : 0;
  *max_per_block_optin =
      context->get_device()->get_gpgpu()->shared_mem_per_block_optin();
  return static_cast<unsigned long long>(*static_smem) + requested_dynamic <=
         *max_per_block_optin;
}

#if CUDART_VERSION >= 9000
/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p entry.
 * The parameter \p entry must be a pointer to a function that executes
 * on the device. The parameter specified by \p entry must be declared as a \p
 * __global__ function. The enumeration defined by \p attr is set to the value
 * defined by \p value If the specified function does not exist, then
 * ::cudaErrorInvalidDeviceFunction is returned. If the specified attribute
 * cannot be written, or if the value is incorrect, then ::cudaErrorInvalidValue
 * is returned.
 *
 * Valid values for \p attr are:
 * ::cuFuncAttrMaxDynamicSharedMem - Maximum size of dynamic shared memory per
 * block
 * ::cudaFuncAttributePreferredSharedMemoryCarveout - Preferred shared memory-L1
 * cache split ratio
 *
 * \param entry - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void
 * **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig
 * (C++ API)", \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const
 * void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(T, size_t) "cudaSetupArgument (C++ API)"
 */
cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func,
                                           enum cudaFuncAttribute attr,
                                           int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  gpgpu_context *ctx = GPGPU_Context();
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel((const char *)func);
  // yasai: a kernel registered without PTX (none loaded for it) is not a valid device function.
  if (entry == NULL) {
    printf("GPGPU-Sim PTX: cudaFuncSetAttribute: no PTX for %s\n",
           context->yasai_pending_name(func));
    return g_last_cudaError = cudaErrorInvalidDeviceFunction;
  }

  if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) {
    if (value < 0) return g_last_cudaError = cudaErrorInvalidValue;
    unsigned static_smem;
    unsigned max_per_block_optin;
    if (!max_dynamic_shared_memory_fits(
            context, entry, (unsigned)value, &static_smem,
            &max_per_block_optin)) {
      printf("GPGPU-Sim PTX: cudaFuncSetAttribute "
             "MaxDynamicSharedMemorySize invalid for '%s': static=%u, "
             "dynamic=%u, max_per_block_optin=%u\n",
             entry->get_name().c_str(), static_smem, (unsigned)value,
             max_per_block_optin);
      return g_last_cudaError = cudaErrorInvalidValue;
    }
    context->get_device()->get_gpgpu()->set_kernel_max_dynamic_smem(
        entry->get_name(), (unsigned)value, static_smem);
    printf("GPGPU-Sim PTX: cudaFuncSetAttribute "
           "MaxDynamicSharedMemorySize for '%s' = %u bytes\n",
           entry->get_name().c_str(), (unsigned)value);
    return g_last_cudaError = cudaSuccess;
  }

  if (attr == cudaFuncAttributePreferredSharedMemoryCarveout) {
    if (value != cudaSharedmemCarveoutDefault &&
        (value < 0 || value > 100)) {
      return g_last_cudaError = cudaErrorInvalidValue;
    }
    context->get_device()->get_gpgpu()->set_kernel_preferred_shared_carveout(
        entry->get_name(), value);
    printf("GPGPU-Sim PTX: cudaFuncSetAttribute "
           "PreferredSharedMemoryCarveout for '%s' = %d\n",
           entry->get_name().c_str(), value);
    return g_last_cudaError = cudaSuccess;
  }

  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s "
         "( func=%p, attr=%d, value=%d )\"\n",
         __my_func__, func, attr, value);
  return g_last_cudaError = cudaSuccess;
}
#endif

cudaError_t CUDARTAPI cudaLibraryLoadData(
    cudaLibrary_t *library, const void *code, cudaJitOption *jitOptions,
    void **jitOptionsValues, unsigned int numJitOptions,
    cudaLibraryOption *libraryOptions, void **libraryOptionValues,
    unsigned int numLibraryOptions) {
  (void)code;
  (void)jitOptions;
  (void)jitOptionsValues;
  (void)numJitOptions;
  (void)libraryOptions;
  (void)libraryOptionValues;
  (void)numLibraryOptions;
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (library == NULL) return g_last_cudaError = cudaErrorInvalidValue;

  CUmodule module = NULL;
  CUresult result = load_module_from_cuda_library_ptx(&module, __my_func__);
  if (result != CUDA_SUCCESS) {
    return g_last_cudaError = cudaErrorInvalidValue;
  }

  *library = (cudaLibrary_t)module;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaLibraryUnload(cudaLibrary_t library) {
  (void)library;
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaLibraryGetKernel(cudaKernel_t *pKernel,
                                           cudaLibrary_t library,
                                           const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (pKernel == NULL || library == NULL || name == NULL) {
    return g_last_cudaError = cudaErrorInvalidValue;
  }

  CUfunction function = NULL;
  CUresult result = cuModuleGetFunction(&function, (CUmodule)library, name);
  if (result != CUDA_SUCCESS) return g_last_cudaError = cudaErrorInvalidValue;

  *pKernel = (cudaKernel_t)function;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaKernelSetAttributeForDevice(
    cudaKernel_t kernel, enum cudaFuncAttribute attr, int value, int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)device;
  return cudaFuncSetAttribute((const void *)kernel, attr, value);
}

extern "C" cudaError_t CUDARTAPI _cudaLibraryLoadData(
    cudaLibrary_t *library, const void *code, cudaJitOption *jitOptions,
    void **jitOptionsValues, unsigned int numJitOptions,
    cudaLibraryOption *libraryOptions, void **libraryOptionValues,
    unsigned int numLibraryOptions) {
  return cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues,
                             numJitOptions, libraryOptions, libraryOptionValues,
                             numLibraryOptions);
}

extern "C" cudaError_t CUDARTAPI _cudaLibraryUnload(cudaLibrary_t library) {
  return cudaLibraryUnload(library);
}

extern "C" cudaError_t CUDARTAPI _cudaLibraryGetKernel(
    cudaKernel_t *pKernel, cudaLibrary_t library, const char *name) {
  return cudaLibraryGetKernel(pKernel, library, name);
}

extern "C" cudaError_t CUDARTAPI _cudaFuncSetAttribute(
    const void *func, enum cudaFuncAttribute attr, int value) {
  return cudaFuncSetAttribute(func, attr, value);
}

extern "C" cudaError_t CUDARTAPI _cudaKernelSetAttributeForDevice(
    cudaKernel_t kernel, enum cudaFuncAttribute attr, int value, int device) {
  return cudaKernelSetAttributeForDevice(kernel, attr, value, device);
}

extern "C" cudaError_t CUDARTAPI _cudaGetDevice(int *device) {
  return cudaGetDevice(device);
}

extern "C" cudaError_t CUDARTAPI _cudaGetDeviceCount(int *count) {
  return cudaGetDeviceCount(count);
}

extern "C" cudaError_t CUDARTAPI _cudaDeviceGetAttribute(
    int *value, enum cudaDeviceAttr attr, int device) {
  return cudaDeviceGetAttribute(value, attr, device);
}

extern "C" cudaError_t CUDARTAPI _cudaSetDevice(int device) {
  return cudaSetDevice(device);
}

cudaError_t CUDARTAPI cudaGLSetGLDevice(int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaErrorUnknown;
}

typedef void *HGPUNV;

cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer,
                                  void *val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
}

namespace cuda_math {

void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer,
                                  void *val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO This function should syncronize if we support Asyn kernel calls
  return g_last_cudaError = cudaSuccess;
}

}  // namespace cuda_math

////////

/// static functions

int cuda_runtime_api::load_static_globals(symbol_table *symtab,
                                          addr_t min_gaddr, gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading globals with explicit initializers... \n");
  fflush(stdout);
  int ng_bytes = 0;
  symbol_table::iterator g = symtab->global_iterator_begin();

  // yasai: every PTX file (and every per-kernel piece) parses into one global table, so each
  // load visits the variables of all earlier loads too. A variable is initialized once, at the
  // load that defines it; before, a later load reset variables that kernels had written.
  static std::set<const symbol *> initialized;
  for (; g != symtab->global_iterator_end(); g++) {
    symbol *global = *g;
    if (global->has_initializer() && initialized.insert(global).second) {
      printf("GPGPU-Sim PTX:     initializing '%s' ... ",
             global->name().c_str());
      addr_t addr = global->get_address();
      const type_info *type = global->type();
      type_info_key ti = type->get_key();
      size_t size;
      int t;
      ti.type_decode(size, t);
      int nbytes = size / 8;
      int offset = 0;
      std::list<operand_info> init_list = global->get_initializer();
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        assert((addr + offset + nbytes) <
               min_gaddr);  // min_gaddr is start of "heap" for cudaMalloc
        gpu->get_global_memory()->write(addr + offset, nbytes, &value, NULL,
                                        NULL);  // assuming little endian here
        offset += nbytes;
        ng_bytes += nbytes;
      }
      printf(" wrote %u bytes\n", offset);
    }
  }
  printf("GPGPU-Sim PTX: finished loading globals (%u bytes total).\n",
         ng_bytes);
  fflush(stdout);
  return ng_bytes;
}

int cuda_runtime_api::load_constants(symbol_table *symtab, addr_t min_gaddr,
                                     gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading constants with explicit initializers... ");
  fflush(stdout);
  int nc_bytes = 0;
  symbol_table::iterator g = symtab->const_iterator_begin();

  // yasai: initialized once, as in load_static_globals.
  static std::set<const symbol *> initialized;
  for (; g != symtab->const_iterator_end(); g++) {
    symbol *constant = *g;
    if (constant->is_const() && constant->has_initializer() && initialized.insert(constant).second) {
      // get the constant element data size
      int basic_type;
      size_t num_bits;
      constant->type()->get_key().type_decode(num_bits, basic_type);

      std::list<operand_info> init_list = constant->get_initializer();
      int nbytes_written = 0;
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        int nbytes = num_bits / 8;
        switch (op.get_type()) {
          case int_t:
            assert(nbytes >= 1);
            break;
          case float_op_t:
            assert(nbytes == 4);
            break;
          case double_op_t:
            assert(nbytes >= 4);
            break;  // account for double DEMOTING
          default:
            abort();
        }
        unsigned addr = constant->get_address() + nbytes_written;
        assert(addr + nbytes < min_gaddr);

        gpu->get_global_memory()->write(
            addr, nbytes, &value, NULL,
            NULL);  // assume little endian (so u8 is the first byte in u32)
        nc_bytes += nbytes;
        nbytes_written += nbytes;
      }
    }
  }
  printf(" done.\n");
  fflush(stdout);
  return nc_bytes;
}

kernel_info_t *cuda_runtime_api::gpgpu_cuda_ptx_sim_init_grid(
    const char *hostFun, gpgpu_ptx_sim_arg_list_t args, struct dim3 gridDim,
    struct dim3 blockDim, CUctx_st *context) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  function_info *entry = context->get_kernel(hostFun);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  /*
  Passing a snapshot of the GPU's current texture mapping to the kernel's info
  as kernels should use texture bindings present at the time of their launch.
  */
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  if (entry == NULL) {
    printf(
        "GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found "
        "for %p\n",
        hostFun);
    abort();
  }
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }

  entry->finalize(result->get_param_memory());
  gpgpu_ctx->func_sim->g_ptx_kernel_count++;
  fflush(stdout);

  if (g_debug_execution >= 4) {
    entry->ptx_jit_config(g_mallocPtr_Size, result->get_param_memory(),
                          (gpgpu_t *)context->get_device()->get_gpgpu(),
                          gridDim, blockDim);
  }

  return result;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
//***extra api for pytorch***

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress,
    const cuuint64_t *globalDim, const cuuint64_t *globalStrides,
    const cuuint32_t *boxDim, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {

  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  // Null pointer checks
  if (!tensorMap || !globalAddress || !globalDim || !globalStrides ||
      !boxDim || !elementStrides)
    return CUDA_ERROR_INVALID_VALUE;

  // tensorRank: must be 1-5
  if (tensorRank < 1 || tensorRank > 5)
    return CUDA_ERROR_INVALID_VALUE;

  // globalAddress: must be 16-byte aligned
  if (reinterpret_cast<uint64_t>(globalAddress) % 16 != 0)
    return CUDA_ERROR_INVALID_VALUE;

  // Per-dimension validation
  for (cuuint32_t i = 0; i < tensorRank; i++) {
    // globalDim: must be non-zero
    if (globalDim[i] == 0)
      return CUDA_ERROR_INVALID_VALUE;
    // boxDim: must be non-zero and <= 256
    if (boxDim[i] == 0 || boxDim[i] > 256)
      return CUDA_ERROR_INVALID_VALUE;
    // elementStrides: must be non-zero and <= 8
    if (elementStrides[i] == 0 || elementStrides[i] > 8)
      return CUDA_ERROR_INVALID_VALUE;
  }

  // globalStrides: must be a multiple of 16 and < 2^40
  for (cuuint32_t i = 0; i < tensorRank - 1; i++) {
    if (globalStrides[i] % 16 != 0)
      return CUDA_ERROR_INVALID_VALUE;
    if (globalStrides[i] >= (1ULL << 40))
      return CUDA_ERROR_INVALID_VALUE;
  }

  // Enum range checks
  if (swizzle > CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B)
    return CUDA_ERROR_INVALID_VALUE;
  if (interleave > CU_TENSOR_MAP_INTERLEAVE_32B)
    return CUDA_ERROR_INVALID_VALUE;

  tensormap_descriptor_t desc;
  memset(&desc, 0, sizeof(desc));

  desc.fields.globalAddress = reinterpret_cast<uint64_t>(globalAddress);
  desc.fields.tensorRank = tensorRank - 1;  // API: 1-based → internal: 0-based

  // Map CUDA API data type enum to internal TMA dtype constants
  switch (tensorDataType) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      desc.fields.tensorDataType = TMA_DTYPE_U8;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
      desc.fields.tensorDataType = TMA_DTYPE_U16;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
      desc.fields.tensorDataType = TMA_DTYPE_U32;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
      desc.fields.tensorDataType = TMA_DTYPE_S32;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
      desc.fields.tensorDataType = TMA_DTYPE_U64;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
      desc.fields.tensorDataType = TMA_DTYPE_S64;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      desc.fields.tensorDataType = TMA_DTYPE_F16;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
      desc.fields.tensorDataType = TMA_DTYPE_F32;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      desc.fields.tensorDataType = TMA_DTYPE_F64;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      desc.fields.tensorDataType = TMA_DTYPE_BF16;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
      desc.fields.tensorDataType = TMA_DTYPE_F32_FTZ;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
      desc.fields.tensorDataType = TMA_DTYPE_TF32;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      desc.fields.tensorDataType = TMA_DTYPE_TF32_FTZ;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B:  // yasai: FP4/FP6 operands of SM100 GEMMs
      desc.fields.tensorDataType = TMA_DTYPE_16U4_ALIGN8B;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B:
      desc.fields.tensorDataType = TMA_DTYPE_16U4_ALIGN16B;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B:
      desc.fields.tensorDataType = TMA_DTYPE_16U6_ALIGN16B;
      break;
    default:
      printf("WARNING: cuTensorMapEncodeTiled: unsupported tensorDataType %d\n",
             (int)tensorDataType);
      return CUDA_ERROR_INVALID_VALUE;
  }

  // When interleave is NONE, boxDim[0] * elementSize must be a multiple of 16
  if (interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) {
    uint32_t elemSize = desc.get_element_size();
    if ((boxDim[0] * elemSize) % 16 != 0) {
      printf("WARNING: cuTensorMapEncodeTiled: boxDim[0]=%u * elemSize=%u = %u "
             "is not a multiple of 16\n",
             boxDim[0], elemSize, boxDim[0] * elemSize);
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  for (cuuint32_t i = 0; i < tensorRank; i++) {
    desc.fields.globalDim[i] = static_cast<uint32_t>(globalDim[i]);
    desc.fields.boxDim[i] = boxDim[i];
    desc.fields.elementStrides[i] = elementStrides[i];
  }
  for (cuuint32_t i = 0; i < tensorRank - 1; i++) {
    desc.fields.globalStrides[i] = globalStrides[i];
  }

  desc.fields.interleave = static_cast<uint32_t>(interleave);
  desc.fields.swizzle = static_cast<uint32_t>(swizzle);
  desc.fields.oobFill = static_cast<uint32_t>(oobFill);

  memcpy(tensorMap, &desc, sizeof(desc));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDriverGetVersion(driverVersion);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  int deviceI = -1;
  cudaError_t e = cudaGetDevice(&deviceI);
  assert(e == cudaSuccess);
  assert(deviceI != -1);
  *device = deviceI;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetDeviceCount(count);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(len >= 10);
  strcpy(name, "GPGPU-Sim");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *bytes = 20000000000;  // dummy value
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
#if (CUDART_VERSION > 5000)
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDeviceGetAttribute(pi, (cudaDeviceAttr)attrib, dev);
  assert(e == cudaSuccess);

  return CUDA_SUCCESS;
}
#endif
CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor,
                                           CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 7000

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 7000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags,
                             CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxCreate_v3(CUcontext *pctx, void *paramsArray,
                                           int numParams, unsigned int flags,
                                           CUdevice dev) {
  (void)paramsArray;
  (void)numParams;
  return cuCtxCreate(pctx, flags, dev);
}

extern "C" CUresult CUDAAPI cuCtxCreate_v4(CUcontext *pctx,
                                           void *ctxCreateParams,
                                           unsigned int flags, CUdevice dev) {
  (void)ctxCreateParams;
  return cuCtxCreate(pctx, flags, dev);
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 7000
CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 7000 */

CUresult CUDAAPI cuCtxSynchronize(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4020
CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("cuModuleLoad: fname=%s\n", fname);

  // Get the current context
  gpgpu_context *ctx = GPGPU_Context();
  if (!ctx) {
    printf("ERROR: No active GPGPU context\n");
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUctx_st *context = GPGPUSim_Context(ctx);

  std::string sidecar_ptx = sidecar_ptx_for_module(fname);
  if (!sidecar_ptx.empty()) {
    return load_module_from_ptx_file(module, sidecar_ptx, __my_func__);
  }

  // Allocate a unique module handle using the same mechanism as
  // cudaRegisterFatBinary
  unsigned module_handle = get_next_fat_bin_handle();
  printf("cuModuleLoad: Allocated module handle %u\n", module_handle);

  // 1. INIT: Extract code using cuobjdump on first load (handle == 1)
  if (module_handle == 1) {
    printf("cuModuleLoad: Initializing cuobjdump (first load)\n");
    ctx->api->cuobjdumpInit();
  }

  // 2. REGISTER: Map handle → fname in fatbinmap
  ctx->api->cuobjdumpRegisterFatBinary(module_handle, fname, context);

  ctx->api->extract_code_using_cuobjdump(fname);

  // 3. PARSE: Load PTX code into context's symbol table
  ctx->cuobjdumpParseBinary(module_handle);
  printf("cuModuleLoad: Parsed binary for module handle %u\n", module_handle);

  // Return the module handle as CUmodule
  *module = (CUmodule)(unsigned long long)module_handle;

  printf("cuModuleLoad: Successfully loaded module with handle %u\n",
         module_handle);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return yasai_load_module_image(module, image, __my_func__);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)numOptions;
  (void)options;
  (void)optionValues;
  return yasai_load_module_image(module, image, __my_func__);
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("cuModuleGetFunction: hmod=%p, name=%s\n", hmod, name);

  // Get the current context
  gpgpu_context *ctx = GPGPU_Context();
  if (!ctx) {
    printf("ERROR: No active GPGPU context\n");
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUctx_st *context = GPGPUSim_Context(ctx);

  // Convert module handle to unsigned
  unsigned module_handle = (unsigned)(unsigned long long)hmod;

  // Look up the symbol table for this module using the public accessor
  symbol_table *symtab = context->get_symbol_table(module_handle);
  if (symtab == NULL) {
    printf("ERROR: Module handle %u not found\n", module_handle);
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // Look up the kernel function by name in the symbol table
  symbol *sym = symtab->lookup(name);
  if (sym == NULL) {
    printf("ERROR: Kernel '%s' not found in module\n", name);
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // Get the function_info from the symbol
  function_info *deviceFunc = sym->get_pc();
  if (deviceFunc == NULL) {
    printf("ERROR: Function info for '%s' is NULL\n", name);
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // ! Use the deviceFunc as the hostFunc for now.
  const char *hostFunc = reinterpret_cast<const char *>(deviceFunc);

  context->register_function(module_handle, hostFunc, name);

  // Return the function as a CUfunction (we use the function_info pointer as
  // the handle)
  *hfunc = (CUfunction)hostFunc;

  printf("cuModuleGetFunction: Found kernel '%s' at %p\n", name, hostFunc);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                   CUmodule hmod, const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                   const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                                    const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6050

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(type == CU_JIT_INPUT_PTX);
  cuda_not_implemented(__my_func__, __LINE__);
  return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  return cuLinkAddFileInternal(state, type, path, numOptions, options,
                               optionValues);
}
#endif

#if CUDART_VERSION >= 5050

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut,
                                size_t *sizeOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // all cuLink* function are implemented to block until completion so nothing
  // to do here
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5050 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: the cudaMalloc allocator.
  void *p = nullptr;
  if (!dptr || cudaMalloc(&p, bytesize) != cudaSuccess) return CUDA_ERROR_OUT_OF_MEMORY;
  *dptr = CUdeviceptr(reinterpret_cast<uint64_t>(p));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaFree(reinterpret_cast<void *>(dptr));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                      CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: cudaMalloc allocations, cuMem mappings and host windows.
  uint64_t base = 0;
  size_t size = 0;
  if (!yasai_vmm::address_range(uint64_t(dptr), &base, &size)) return CUDA_ERROR_NOT_FOUND;
  if (pbase) *pbase = CUdeviceptr(base);
  if (psize) *psize = size;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuMemFreeHost(void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize,
                                unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6000

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                   unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 6000 */

#if CUDART_VERSION >= 4010

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: forwards to the runtime IPC functions.
  return cudaIpcGetEventHandle(reinterpret_cast<cudaIpcEventHandle_t *>(pHandle),
                               reinterpret_cast<cudaEvent_t>(event)) == cudaSuccess
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent,
                                      CUipcEventHandle handle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: forwards to the runtime IPC functions.
  cudaIpcEventHandle_t h;
  memcpy(&h, &handle, sizeof(h));
  return cudaIpcOpenEventHandle(reinterpret_cast<cudaEvent_t *>(phEvent), h) == cudaSuccess
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: forwards to the runtime IPC functions.
  return cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t *>(pHandle),
                             reinterpret_cast<void *>(dptr)) == cudaSuccess
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: forwards to the runtime IPC functions.
  cudaIpcMemHandle_t h;
  memcpy(&h, &handle, sizeof(h));
  void *p = nullptr;
  if (!pdptr || cudaIpcOpenMemHandle(&p, h, Flags) != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;
  *pdptr = CUdeviceptr(reinterpret_cast<uint64_t>(p));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: forwards to the runtime IPC functions.
  return cudaIpcCloseMemHandle(reinterpret_cast<void *>(dptr)) == cudaSuccess
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

#endif /* CUDART_VERSION >= 4010 */

#if CUDART_VERSION >= 6050
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size,
                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)flags;
  // yasai: registered host memory is mapped in place, whatever the flags (the simulator has no
  // separate device copy to keep coherent).
  if (ptr == NULL || size == 0) return g_last_cudaError = cudaErrorInvalidValue;
  host_window::add(reinterpret_cast<uint64_t>(ptr), size);
  GPGPU_Context()->api->pinned_memory_size[ptr] = size;
  return g_last_cudaError = cudaSuccess;
}

extern "C" __host__ cudaError_t cudaProfilerStart() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return g_last_cudaError = cudaSuccess;
}

extern "C" __host__ cudaError_t cudaProfilerStop() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return g_last_cudaError = cudaSuccess;
}

#endif
#if CUDART_VERSION >= 4000

CUresult CUDAAPI cuMemHostUnregister(void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                              CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                              size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                              const void *srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                              CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                   const void *srcHost, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                              unsigned char uc, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned short us, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned int ui, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayCreate(CUarray *pHandle,
                               const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                      CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuArray3DCreate(
    CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 5000

CUresult CUDAAPI
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

/** @} */ /* END CUDA_MEM */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuPointerGetAttribute(void *data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: answered from the allocation, mapping and host window tables.
  return CUresult(yasai_vmm::pointer_attribute(data, int(attribute), uint64_t(ptr),
                                               GPGPUSim_Context(GPGPU_Context())));
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 8000
__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(
    cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc,
    const cudaTextureDesc *pTexDesc, const cudaResourceViewDesc *pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                         CUmem_range_attribute *attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 8000 */

#if CUDART_VERSION >= 6000
CUresult CUDAAPI cuPointerSetAttribute(const void *value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 6000 */

#if CUDART_VERSION >= 7000
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute *attributes,
                                        void **data, CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // yasai: attributes that do not apply leave their value untouched, as the driver does.
  for (unsigned i = 0; i < numAttributes; i++)
    if (data && data[i])
      yasai_vmm::pointer_attribute(data[i], int(attributes[i]), uint64_t(ptr),
                                   GPGPUSim_Context(GPGPU_Context()));
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 7000 */

/** @} */ /* END CUDA_UNIFIED */

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                            unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6000

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 6000 */

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

/** @} */ /* END CUDA_STREAM */

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 8000
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 8000 */

/** @} */ /* END CUDA_EVENT */

// yasai: function attributes from the kernel's ptxas information and the device configuration
// (CUfunction values are function_info pointers). Triton reads NUM_REGS, LOCAL_SIZE_BYTES and
// MAX_DYNAMIC_SHARED_SIZE_BYTES for every kernel it loads; attributes the simulator has no
// meaning for (cache mode, cluster preferences) read 0.
//   pi: receives the value; attrib: attribute; hfunc: function
CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (pi == NULL) return CUDA_ERROR_INVALID_VALUE;
  function_info *entry = reinterpret_cast<function_info *>(hfunc);
  if (entry == NULL) return CUDA_ERROR_INVALID_HANDLE;
  gpgpu_context *ctx = GPGPU_Context();
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_sim *gpu = context->get_device()->get_gpgpu();
  const struct gpgpu_ptx_sim_info *k = entry->get_kernel_info();
  const struct cudaDeviceProp *prop = context->get_device()->get_prop();
  switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: {
      struct cudaFuncAttributes fa;
      memset(&fa, 0, sizeof(fa));
      fa.numRegs = k->regs;
      fa.sharedSizeBytes = k->smem;
      *pi = k->maxthreads > 0 ? (int)k->maxthreads : (int)getMaxThreadsPerBlock(&fa, ctx);
      break;
    }
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: *pi = (int)k->smem; break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: *pi = (int)k->cmem; break;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: *pi = (int)k->lmem; break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: *pi = (int)k->regs; break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: *pi = (int)k->ptx_version; break;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *pi = (int)k->sm_target; break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: {
      const unsigned set = gpu->has_kernel_max_dynamic_smem(entry->get_name())
                               ? gpu->get_kernel_max_dynamic_smem(entry->get_name())
                               : 0;
      const unsigned optin = (unsigned)prop->sharedMemPerBlockOptin;
      *pi = set != 0 ? (int)set : (int)(optin > k->smem ? optin - k->smem : 0);
      break;
    }
    default: *pi = 0; break;
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                                      CUkernel kernel, CUdevice dev) {
  (void)dev;
  return cuFuncGetAttribute(pi, attrib, (CUfunction)kernel);
}

extern "C" CUresult CUDAAPI _cuKernelGetAttribute(
    int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) {
  return cuKernelGetAttribute(pi, attrib, kernel, dev);
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {

  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  function_info *entry = reinterpret_cast<function_info *>(hfunc);
  if (entry == NULL) return CUDA_ERROR_INVALID_HANDLE;

  gpgpu_context *ctx = GPGPU_Context();
  CUctx_st *context = GPGPUSim_Context(ctx);

  if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
    if (value < 0) return CUDA_ERROR_INVALID_VALUE;
    unsigned static_smem;
    unsigned max_per_block_optin;
    if (!max_dynamic_shared_memory_fits(
            context, entry, (unsigned)value, &static_smem,
            &max_per_block_optin)) {
      printf("GPGPU-Sim PTX: cuFuncSetAttribute "
             "MAX_DYNAMIC_SHARED_SIZE_BYTES invalid for '%s': static=%u, "
             "dynamic=%u, max_per_block_optin=%u\n",
             entry->get_name().c_str(), static_smem, (unsigned)value,
             max_per_block_optin);
      return CUDA_ERROR_INVALID_VALUE;
    }
    context->get_device()->get_gpgpu()->set_kernel_max_dynamic_smem(
        entry->get_name(), (unsigned)value, static_smem);
    printf("GPGPU-Sim PTX: cuFuncSetAttribute "
           "MAX_DYNAMIC_SHARED_SIZE_BYTES for '%s' = %u bytes\n",
           entry->get_name().c_str(), (unsigned)value);
    return CUDA_SUCCESS;
  }

  if (attrib == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
    if (value != CU_SHAREDMEM_CARVEOUT_DEFAULT &&
        (value < 0 || value > 100)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    context->get_device()->get_gpgpu()->set_kernel_preferred_shared_carveout(
        entry->get_name(), value);
    printf("GPGPU-Sim PTX: cuFuncSetAttribute "
           "PREFERRED_SHARED_MEMORY_CARVEOUT for '%s' = %d\n",
           entry->get_name().c_str(), value);
    return CUDA_SUCCESS;
  }

  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s "
         "( func=%p, attrib=%d, value=%d )\"\n",
         __my_func__, hfunc, attrib, value);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4020
CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  return cuLaunchKernelInternal(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                blockDimY, blockDimZ, sharedMemBytes, hStream,
                                kernelParams, extra);
}
#endif /* CUDART_VERSION >= 4000 */

/** @} */ /* END CUDA_EXEC */

CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void *ptr,
                             unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunch(CUfunction f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width,
                                   int grid_height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit,
                                  CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
/** @} */ /* END CUDA_EXEC_DEPRECATED */

#if CUDART_VERSION >= 6050

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_OCCUPANCY */
#endif    /* CUDART_VERSION >= 6050 */

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR *desc,
                                      CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                           CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                        int dim) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                                   CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                             float *pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_SURFREF */

#if CUDART_VERSION >= 5000
CUresult CUDAAPI
cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                  const CUDA_TEXTURE_DESC *pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                            CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                           CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_TEXOBJECT */

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                                    const CUDA_RESOURCE_DESC *pResDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                             CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_PEER_ACCESS */
#endif    /* CUDART_VERSION >= 4000 */

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 5000

CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_GRAPHICS */

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetExportTable(ppExportTable, pExportTableId);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 4000 && CUDART_VERSION < 6050)
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* defined(CUDART_VERSION_INTERNAL) || (CUDART_VERSION >= 4000 && \
          CUDART_VERSION < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 5050 && CUDART_VERSION < 6050)
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 5050 && CUDART_VERSION \
          < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 3020 && CUDART_VERSION < 4010)
CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef,
                                         const CUDA_ARRAY_DESCRIPTOR *desc,
                                         CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 3020 && CUDART_VERSION \
          < 4010) */

#if defined(CUDART_VERSION_INTERNAL) || CUDART_VERSION < 4000
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || CUDART_VERSION < 4000 */

#if defined(CUDART_VERSION_INTERNAL)
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset,
                                 const void *srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray,
                                      size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                      CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc,
                               size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                 unsigned char uc, size_t Width,
                                 size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned short us, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned int ui, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
#endif

CUresult cuProfilerInitialize(const char *configFile, const char *outputFile,
                              CUoutput_mode outputMode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult cuProfilerStart(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
CUresult cuProfilerStop(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

//_ptds

extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                          size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeer_ptds(CUdeviceptr dstDevice,
                                              CUcontext dstContext,
                                              CUdeviceptr srcDevice,
                                              CUcontext srcContext,
                                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 const void *srcHost,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void *dstHost,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr dstDevice,
                                               unsigned char uc,
                                               unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned short us,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned int ui,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8_v2_ptds(CUdeviceptr dstDevice,
                                                 unsigned int dstPitch,
                                                 unsigned char uc,
                                                 unsigned int Width,
                                                 unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D16_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned short us,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D32_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned int ui,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

//_ptsz
extern "C" CUresult CUDAAPI
cuMemcpy3DPeer_ptsz(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                               size_t ByteCount,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeerAsync_ptsz(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
    CUcontext srcContext, size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray,
                                                      size_t dstOffset,
                                                      const void *srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyAtoHAsync_v2_ptsz(void *dstHost,
                                                      CUarray srcArray,
                                                      size_t srcOffset,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      const void *srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void *dstHost,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr dstDevice,
                                                 unsigned char uc, size_t N,
                                                 CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice,
                                                   size_t dstPitch,
                                                   unsigned char uc,
                                                   size_t Width, size_t Height,
                                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent hEvent,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWriteValue32_ptsz(CUstream stream,
                                                      CUdeviceptr addr,
                                                      cuuint32_t value,
                                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWaitValue32_ptsz(CUstream stream,
                                                     CUdeviceptr addr,
                                                     cuuint32_t value,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamBatchMemOp_ptsz(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetPriority_ptsz(CUstream hStream,
                                                     int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetFlags_ptsz(CUstream hStream,
                                                  unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream hStream,
                                                   CUevent hEvent,
                                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamAddCallback_ptsz(CUstream hStream,
                                                     CUstreamCallback callback,
                                                     void *userData,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamAttachMemAsync_ptsz(CUstream hStream,
                                                        CUdeviceptr dptr,
                                                        size_t length,
                                                        unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsMapResources_ptsz(
    unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsUnmapResources_ptsz(
    unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr,
                                                    size_t count,
                                                    CUdevice dstDevice,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet %s\n", __my_func__);
  return CUDA_SUCCESS;
}

extern "C" {

#define cuda_error_not_impl \
  if (g_debug_execution >= 3) { \
    announce_call(__my_func__); \
  } \
  printf("ERROR: %s is not implemented yet.", __my_func__); \
  abort(); \
  return cudaSuccess;

__host__ cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
  // yasai: stream capture is not simulated; keep a per-thread mode so swap pairs round-trip.
  static thread_local enum cudaStreamCaptureMode t_mode = cudaStreamCaptureModeGlobal;
  if (mode == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  enum cudaStreamCaptureMode old = t_mode;
  t_mode = *mode;
  *mode = old;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) {
  // yasai: pools are tokens (see cudaMemPoolCreate); the default pool is token 0x1000 + device.
  if (memPool == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  *memPool = reinterpret_cast<cudaMemPool_t>(uintptr_t(0x1000 + device));
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ) {
  // yasai: pool attributes are not modeled; every attribute reads as zero.
  (void)memPool;
  if (value == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  if (attr == cudaMemPoolReuseFollowEventDependencies || attr == cudaMemPoolReuseAllowOpportunistic ||
      attr == cudaMemPoolReuseAllowInternalDependencies)
    *static_cast<int *>(value) = 0;
  else
    *static_cast<unsigned long long *>(value) = 0;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ) {
  // yasai: pool attributes are not modeled; setting one is accepted and has no effect.
  (void)memPool;
  (void)attr;
  (void)value;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count) {
  // yasai: every allocation is accessible from the single simulated device.
  (void)memPool;
  (void)descList;
  (void)count;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
  // yasai: pools hold no cached memory, so there is nothing to trim.
  (void)memPool;
  (void)minBytesToKeep;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
  // yasai: no stream is ever capturing in the simulator.
  (void)stream;
  if (pCaptureStatus) *pCaptureStatus = cudaStreamCaptureStatusNone;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
  // yasai: stream-ordered allocation is an ordinary allocation.
  (void)hStream;
  return cudaMalloc(devPtr, size);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0)) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
  // yasai: stream-ordered free is an ordinary free.
  (void)hStream;
  return cudaFree(devPtr);
}

// yasai: an opened IPC event shares the exporter's counters, so cudaStreamWaitEvent and
// cudaEventSynchronize in this process wait for records made in the other.
__host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
  unsigned int *counters = nullptr;
  char ref[64];
  if (!event || !yasai_vmm::ipc_event_open(&handle, &counters, ref))
    return g_last_cudaError = cudaErrorInvalidValue;
  CUevent_st *e = new CUevent_st(false);
  e->set_shared(counters, ref);
  g_timer_events[e->get_uid()] = e;
  *event = e;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
  CUevent_st *e = get_event(event);
  if (!handle || !e || !e->shared_name()[0]) return g_last_cudaError = cudaErrorInvalidResourceHandle;
  yasai_vmm::ipc_event_handle(handle, e->shared_name());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr) {
  if (!yasai_vmm::ipc_close_mem_handle(reinterpret_cast<uint64_t>(devPtr)))
    return g_last_cudaError = cudaErrorInvalidValue;
  return g_last_cudaError = cudaSuccess;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device) {
  return cudaGetDevicePropertiesInternal(prop, device);
}

__host__ cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0)) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0)) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(
    cudaGraphNode_t node, cudaGraphNode_t *pDependencies,
    size_t *pNumDependencies) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaUserObjectCreate(
    cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy,
    unsigned int initialRefcount, unsigned int flags) {
  cuda_error_not_impl;
}

typedef void (CUDART_CB *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);

__host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
  cuda_error_not_impl;
}

__host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), cudaGraph_t *graph_out __dv(0), const cudaGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0)) {
  // yasai: no stream is ever capturing in the simulator.
  (void)stream;
  if (captureStatus_out) *captureStatus_out = cudaStreamCaptureStatusNone;
  if (id_out) *id_out = 0;
  if (graph_out) *graph_out = NULL;
  if (dependencies_out) *dependencies_out = NULL;
  if (numDependencies_out) *numDependencies_out = 0;
  return g_last_cudaError = cudaSuccess;
}


// yasai: looks a function up in this shared library itself (not in a real libcuda).
// @param symbol Driver function name, for example "cuLaunchKernelEx".
// @return Address, or NULL when neither the name nor its _v2.._v4 variants are defined here.
static void *yasai_find_own_symbol(const char *symbol) {
  static void *self = NULL;
  if (self == NULL) {
    Dl_info info;
    if (dladdr((void *)&yasai_find_own_symbol, &info) && info.dli_fname)
      self = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
  }
  if (self == NULL) return NULL;
  const char *suffixes[] = {"", "_v2", "_v3", "_v4"};
  for (const char *suf : suffixes) {
    std::string n = std::string(symbol) + suf;
    if (void *fp = dlsym(self, n.c_str())) return fp;
  }
  return NULL;
}

__host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(
    const char *symbol, void **funcPtr, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  (void)flags;
  if (!symbol || !funcPtr) {
    if (driverStatus) {
      *driverStatus = cudaDriverEntryPointSymbolNotFound;
    }
    return cudaErrorInvalidValue;
  }

  if (strcmp(symbol, "cuDeviceGet") == 0 ||
      strcmp(symbol, "cuKernelGetAttribute") == 0) {
    *funcPtr = strcmp(symbol, "cuDeviceGet") == 0
                   ? reinterpret_cast<void *>(&cuDeviceGet)
                   : reinterpret_cast<void *>(&cuKernelGetAttribute);
    if (driverStatus) {
      *driverStatus = cudaDriverEntryPointSuccess;
    }
    return cudaSuccess;
  }

  if (strcmp(symbol, "cuTensorMapEncodeTiled") == 0) {
    *funcPtr = reinterpret_cast<void *>(&cuTensorMapEncodeTiled);
    if (driverStatus) {
      *driverStatus = cudaDriverEntryPointSuccess;
    }
    return cudaSuccess;
  }

  // yasai: any driver function this library defines is handed out (NCCL loads its driver
  // calls this way); the _v2 / _v3 names that cuda.h maps base names to are tried as well.
  if (void *fp = yasai_find_own_symbol(symbol)) {
    *funcPtr = fp;
    if (driverStatus) *driverStatus = cudaDriverEntryPointSuccess;
    return cudaSuccess;
  }
  *funcPtr = NULL;
  if (driverStatus) {
    *driverStatus = cudaDriverEntryPointSymbolNotFound;
  }
  return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char *symbol, void **funcPtr, unsigned int cudaVersion,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  if (!symbol || !funcPtr) {
    if (driverStatus) {
      *driverStatus = cudaDriverEntryPointSymbolNotFound;
    }
    return cudaErrorInvalidValue;
  }

  if (strcmp(symbol, "cuTensorMapEncodeTiled") == 0 && cudaVersion < 12000) {
    *funcPtr = NULL;
    if (driverStatus) {
      *driverStatus = cudaDriverEntryPointVersionNotSufficent;
    }
    return cudaSuccess;
  }

  return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
}

__host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) {
  // yasai: undoes cudaHostRegister.
  if (!host_window::remove(reinterpret_cast<uint64_t>(ptr)))
    return g_last_cudaError = cudaErrorHostMemoryNotRegistered;
  GPGPU_Context()->api->pinned_memory_size.erase(ptr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernelExC(
    const cudaLaunchConfig_t *config, const void *func, void **args) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (config == NULL || func == NULL) {
    return g_last_cudaError = cudaErrorInvalidValue;
  }
  // yasai: the cluster shape attribute applies to this launch (see g_yasai_pending_cluster).
  for (unsigned i = 0; i < config->numAttrs; ++i)
    if (config->attrs[i].id == cudaLaunchAttributeClusterDimension)
      g_yasai_pending_cluster = dim3(config->attrs[i].val.clusterDim.x,
                                     config->attrs[i].val.clusterDim.y,
                                     config->attrs[i].val.clusterDim.z);

  cudaConfigureCallInternal(config->gridDim, config->blockDim,
                            config->dynamicSmemBytes, config->stream);
  return cudaLaunchKernelInternal((const char *)func, config->gridDim,
                                  config->blockDim, (const void **)args,
                                  config->dynamicSmemBytes, config->stream);
}

__host__ cudaError_t CUDARTAPI _cudaLaunchKernelEx(
    const cudaLaunchConfig_t *config, const void *func, void **args) {
  return cudaLaunchKernelExC(config, func, args);
}

__host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
  // yasai: runs the host function after the stream's earlier work, in the calling thread.
  if (fn == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  cudaError_t e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) return e;
  fn(userData);
  return g_last_cudaError = cudaSuccess;
}

#undef cuda_error_not_impl

}


// ---- yasai additions for NCCL 2.30.7 -----------------------------------------------------

// yasai: CUfunction values in this simulator are the host stub pointers, so the stub is the
// function handle.
extern "C" __host__ cudaError_t CUDARTAPI cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                                             const void *symbolPtr) {
  if (functionPtr == NULL || symbolPtr == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  *functionPtr = (cudaFunction_t)symbolPtr;
  return g_last_cudaError = cudaSuccess;
}

// yasai: stream id from the simulator's stream uid (0 for the default stream).
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamGetId(cudaStream_t hStream,
                                                         unsigned long long *streamId) {
  if (streamId == NULL) return g_last_cudaError = cudaErrorInvalidValue;
  *streamId = hStream ? (unsigned long long)((struct CUstream_st *)hStream)->get_uid() : 0;
  return g_last_cudaError = cudaSuccess;
}

// yasai: CUDA 13 form of cudaStreamGetCaptureInfo; no stream is ever capturing.
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(
    cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out, const cudaGraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  (void)stream;
  if (captureStatus_out) *captureStatus_out = cudaStreamCaptureStatusNone;
  if (id_out) *id_out = 0;
  if (graph_out) *graph_out = NULL;
  if (dependencies_out) *dependencies_out = NULL;
  if (edgeData_out) *edgeData_out = NULL;
  if (numDependencies_out) *numDependencies_out = 0;
  return g_last_cudaError = cudaSuccess;
}

// yasai: layout of CUlaunchConfig in CUDA 13's cuda.h (CUlaunchConfig_st), which the headers
// bundled with this simulator do not declare.
struct yasai_CUlaunchConfig {
  unsigned int gridDimX;        ///< Grid width in blocks.
  unsigned int gridDimY;        ///< Grid height in blocks.
  unsigned int gridDimZ;        ///< Grid depth in blocks.
  unsigned int blockDimX;       ///< Block x dimension.
  unsigned int blockDimY;       ///< Block y dimension.
  unsigned int blockDimZ;       ///< Block z dimension.
  unsigned int sharedMemBytes;  ///< Dynamic shared memory per block in bytes.
  CUstream hStream;             ///< Stream.
  void *attrs;                  ///< Launch attributes (yasai_CUlaunchAttribute array).
  unsigned int numAttrs;        ///< Number of attributes.
};

// yasai: layout of CUlaunchAttribute (id, padding to 8 bytes, 64-byte value union).
struct yasai_CUlaunchAttribute {
  int id;              ///< CUlaunchAttributeID; 4 is CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION.
  char pad[4];         ///< Padding of the id to 8 bytes.
  unsigned value[16];  ///< Value union; the cluster shape is value[0..2].
};

// yasai: cuLaunchKernelEx; the cluster shape attribute is honored, the other attributes
// (memory-sync domain, priority and so on) are ignored; grid, block, shared memory and stream
// come from the config.
extern "C" CUresult CUDAAPI cuLaunchKernelEx(const yasai_CUlaunchConfig *config, CUfunction f,
                                             void **kernelParams, void **extra) {
  if (config == NULL) return CUDA_ERROR_INVALID_VALUE;
  // yasai: the cluster shape attribute applies to this launch.
  const yasai_CUlaunchAttribute *attrs = (const yasai_CUlaunchAttribute *)config->attrs;
  for (unsigned i = 0; attrs != NULL && i < config->numAttrs; ++i)
    if (attrs[i].id == 4)
      g_yasai_pending_cluster = dim3(attrs[i].value[0], attrs[i].value[1], attrs[i].value[2]);
  return cuLaunchKernelInternal(f, config->gridDimX, config->gridDimY, config->gridDimZ,
                                config->blockDimX, config->blockDimY, config->blockDimZ,
                                config->sharedMemBytes, config->hStream, kernelParams, extra);
}

// yasai: a fixed UUID per simulated device (NCCL reads it for topology and logging).
extern "C" CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  if (uuid == NULL) return CUDA_ERROR_INVALID_VALUE;
  memset(uuid->bytes, 0, sizeof(uuid->bytes));
  uuid->bytes[0] = 0x59;  // 'Y'
  // yasai: the UUID carries the device's PCI bus (YASAI_PCI_BUS_BASE + dev, as in
  // cudaDeviceGetPCIBusId), so ranks on one host that set different bases have different GPUs.
  const char *base_env = getenv("YASAI_PCI_BUS_BASE");
  const unsigned base = base_env ? unsigned(strtoul(base_env, NULL, 0)) : 0xd0u;
  uuid->bytes[14] = (char)((base + dev) & 0xff);
  uuid->bytes[15] = (char)(dev & 0xff);
  return CUDA_SUCCESS;
}
