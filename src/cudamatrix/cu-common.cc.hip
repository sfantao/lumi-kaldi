// cudamatrix/cu-common.cc

// Copyright      2013  Karel Vesely
//                2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#if HAVE_CUDA

#include "cudamatrix/cu-common.h"

#include <hip/hip_runtime.h>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrixdim.h"

namespace kaldi {

#ifdef USE_NVTX
NvtxTracer::NvtxTracer(const char* name) {
  const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
  const int num_colors = sizeof(colors)/sizeof(uint32_t);
  int color_id = ((int)name[0])%num_colors;
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = colors[color_id];
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;
	nvtxRangePushEx(&eventAttrib);
  // nvtxRangePushA(name);
}
NvtxTracer::~NvtxTracer() {
  nvtxRangePop();
}
#endif

hipblasOperation_t KaldiTransToCuTrans(MatrixTransposeType kaldi_trans) {
  hipblasOperation_t cublas_trans;

  if (kaldi_trans == kNoTrans)
    cublas_trans = HIPBLAS_OP_N;
  else if (kaldi_trans == kTrans)
    cublas_trans = HIPBLAS_OP_T;
  else
    cublas_trans = HIPBLAS_OP_C;
  return cublas_trans;
}

void GetBlockSizesForSimpleMatrixOperation(int32 num_rows,
                                           int32 num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock) {
  KALDI_ASSERT(num_rows > 0 && num_cols > 0);
  int32 col_blocksize = 64, row_blocksize = 4;
  while (col_blocksize > 1 &&
         (num_cols + (num_cols / 2) <= col_blocksize ||
          num_rows > 65535 * row_blocksize)) {
    col_blocksize /= 2;
    row_blocksize *= 2;
  }

  dimBlock->x = col_blocksize;
  dimBlock->y = row_blocksize;
  dimBlock->z = 1;
  dimGrid->x = n_blocks(num_cols, col_blocksize);
  dimGrid->y = n_blocks(num_rows, row_blocksize);
  KALDI_ASSERT(dimGrid->y <= 65535 &&
               "Matrix has too many rows to process");
  dimGrid->z = 1;
}

const char* cublasGetStatusStringK(hipblasStatus_t status) {
  // Defined in CUDA include file: hipblas.h or cublas_api.h
  switch(status) {
    case HIPBLAS_STATUS_SUCCESS:           return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:   return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:     return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:     return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:     return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:  return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:    return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:     return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_UNKNOWN:     return "HIPBLAS_STATUS_UNKNOWN";
  }
  return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

const char* cusparseGetStatusString(hipsparseStatus_t status) {
  // detail info come from http://docs.nvidia.com/cuda/cusparse/index.html#cusparsestatust
  // Defined in CUDA include file: hipsparse.h
  switch(status) {
    case HIPSPARSE_STATUS_SUCCESS:                   return "HIPSPARSE_STATUS_SUCCESS";
    case HIPSPARSE_STATUS_NOT_INITIALIZED:           return "HIPSPARSE_STATUS_NOT_INITIALIZED";
    case HIPSPARSE_STATUS_ALLOC_FAILED:              return "HIPSPARSE_STATUS_ALLOC_FAILED";
    case HIPSPARSE_STATUS_INVALID_VALUE:             return "HIPSPARSE_STATUS_INVALID_VALUE";
    case HIPSPARSE_STATUS_ARCH_MISMATCH:             return "HIPSPARSE_STATUS_ARCH_MISMATCH";
    case HIPSPARSE_STATUS_MAPPING_ERROR:             return "HIPSPARSE_STATUS_MAPPING_ERROR";
    case HIPSPARSE_STATUS_EXECUTION_FAILED:          return "HIPSPARSE_STATUS_EXECUTION_FAILED";
    case HIPSPARSE_STATUS_INTERNAL_ERROR:            return "HIPSPARSE_STATUS_INTERNAL_ERROR";
    case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case HIPSPARSE_STATUS_ZERO_PIVOT:                return "HIPSPARSE_STATUS_ZERO_PIVOT";
    #if CUDA_VERSION >= 11000
    case HIPSPARSE_STATUS_NOT_SUPPORTED:             return "HIPSPARSE_STATUS_NOT_SUPPORTED";
    case HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES:    return "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    #endif
  }
  return "CUSPARSE_STATUS_UNKNOWN_ERROR";
}

const char* curandGetStatusString(hiprandStatus_t status) {
  // detail info come from http://docs.nvidia.com/cuda/hiprand/group__HOST.html
  // Defined in CUDA include file: hiprand.h
  switch(status) {
    case HIPRAND_STATUS_SUCCESS:                     return "HIPRAND_STATUS_SUCCESS";
    case HIPRAND_STATUS_VERSION_MISMATCH:            return "HIPRAND_STATUS_VERSION_MISMATCH";
    case HIPRAND_STATUS_NOT_INITIALIZED:             return "HIPRAND_STATUS_NOT_INITIALIZED";
    case HIPRAND_STATUS_ALLOCATION_FAILED:           return "HIPRAND_STATUS_ALLOCATION_FAILED";
    case HIPRAND_STATUS_TYPE_ERROR:                  return "HIPRAND_STATUS_TYPE_ERROR";
    case HIPRAND_STATUS_OUT_OF_RANGE:                return "HIPRAND_STATUS_OUT_OF_RANGE";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:         return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:   return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case HIPRAND_STATUS_LAUNCH_FAILURE:              return "HIPRAND_STATUS_LAUNCH_FAILURE";
    case HIPRAND_STATUS_PREEXISTING_FAILURE:         return "HIPRAND_STATUS_PREEXISTING_FAILURE";
    case HIPRAND_STATUS_INITIALIZATION_FAILED:       return "HIPRAND_STATUS_INITIALIZATION_FAILED";
    case HIPRAND_STATUS_ARCH_MISMATCH:               return "HIPRAND_STATUS_ARCH_MISMATCH";
    case HIPRAND_STATUS_INTERNAL_ERROR:              return "HIPRAND_STATUS_INTERNAL_ERROR";
  }
  return "CURAND_STATUS_UNKNOWN_ERROR";
}

}  // namespace kaldi

#endif  // HAVE_CUDA
