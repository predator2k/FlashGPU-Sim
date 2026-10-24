// yasai: the driver's private export tables ("dark API") that cuGetExportTable hands out. A CUDA
// runtime linked statically into a library (for example libcute_dsl_runtime.so of the CUTLASS
// DSL, which FlashInfer's CuTe DSL kernels use) asks the driver for them during its
// initialization and calls their entries; see yasai_dark_api.cc.
#ifndef YASAI_DARK_API_H
#define YASAI_DARK_API_H

#include "cuda_api.h"

namespace yasai_dark_api {

/// Looks an export table up by its UUID and logs the request (to stderr, unbuffered).
/// @param table Receives the table: an array of entry pointers, entry 0 often the table size.
/// @param id UUID of the table.
/// @return CUDA_SUCCESS for a known table, CUDA_ERROR_INVALID_VALUE for an unknown one (the
///         runtime then does without it, as with ZLUDA), CUDA_ERROR_INVALID_VALUE for a null
///         argument.
CUresult get_export_table(const void **table, const CUuuid *id);

}  // namespace yasai_dark_api

#endif  // YASAI_DARK_API_H
