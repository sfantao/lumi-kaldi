// cudamatrix/cu-common.h

// Copyright 2009-2011  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_CUDAMATRIX_CU_COMMON_H_
#define KALDI_CUDAMATRIX_CU_COMMON_H_

#include <iostream>
#include <sstream>

#include "base/kaldi-error.h"
#include "cudamatrix/cu-matrixdim.h" // for CU1DBLOCK and CU2DBLOCK
#include "matrix/matrix-common.h"

#if HAVE_CUDA

typedef enum {
  CUBLAS_COMPUTE_16F               = 64, /* half - default */
  CUBLAS_COMPUTE_16F_PEDANTIC      = 65, /* half - pedantic */
  CUBLAS_COMPUTE_32F               = 68, /* float - default */
  CUBLAS_COMPUTE_32F_PEDANTIC      = 69, /* float - pedantic */
  CUBLAS_COMPUTE_32F_FAST_16F      = 74, /* float - fast, allows down-converting inputs to half or TF32 */
  CUBLAS_COMPUTE_32F_FAST_16BF     = 75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
  CUBLAS_COMPUTE_32F_FAST_TF32     = 77, /* float - fast, allows down-converting inputs to TF32 */
  CUBLAS_COMPUTE_64F               = 70, /* double - default */
  CUBLAS_COMPUTE_64F_PEDANTIC      = 71, /* double - pedantic */
  CUBLAS_COMPUTE_32I               = 72, /* signed 32-bit int - default */
  CUBLAS_COMPUTE_32I_PEDANTIC      = 73, /* signed 32-bit int - pedantic */
} hipblasComputeType_t;

#include <hipblas.h>
#include <hip/hip_runtime_api.h>
#include <hiprand.h>
#include <hipsparse.h>
#include <nvToolsExt.h>

#define CU_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "hipError_t " << ret << " : \"" << hipGetErrorString((hipError_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define CUFFT_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != HIPFFT_SUCCESS) { \
    KALDI_ERR << "hipfftResult " << ret << " returned from '" << #fun << "'"; \
  } \
}

#define CUBLAS_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "hipblasStatus_t " << ret << " : \"" << cublasGetStatusStringK((hipblasStatus_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define CUSOLVER_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "cusolverStatus_t " << ret << " : \"" << ret << "\" returned from '" << #fun << "'"; \
  } \
}


#define CUSPARSE_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "hipsparseStatus_t " << ret << " : \"" << cusparseGetStatusString((hipsparseStatus_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define CURAND_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "hiprandStatus_t " << ret << " : \"" << curandGetStatusString((hiprandStatus_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define KALDI_CUDA_ERR(ret, msg) \
{ \
  if (ret != 0) { \
    KALDI_ERR << msg << ", diagnostics: hipError_t " << ret << " : \"" << hipGetErrorString((hipError_t)ret) << "\", in " << __FILE__ << ":" << __LINE__; \
  } \
}


namespace kaldi {

#ifdef USE_NVTX
class NvtxTracer {
public:
    NvtxTracer(const char* name);
    ~NvtxTracer();
};
#define NVTX_RANGE(name) NvtxTracer uniq_name_using_macros(name);
#else
#define NVTX_RANGE(name)
#endif

/** Number of blocks in which the task of size 'size' is splitted **/
inline int32 n_blocks(int32 size, int32 block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

hipblasOperation_t KaldiTransToCuTrans(MatrixTransposeType kaldi_trans);


/*
  This function gives you suitable dimBlock and dimGrid sizes for a simple
  matrix operation (one that applies to each element of the matrix.  The x
  indexes will be interpreted as column indexes, and the y indexes will be
  interpreted as row indexes; this is based on our interpretation of a matrix as
  being row-major, i.e.  having column-stride = 1, not based on CuBLAS's
  opposite interpretation.  There is a good reason for associating the column
  index with x and not y; this helps memory locality in adjacent kernels.
 */
void GetBlockSizesForSimpleMatrixOperation(int32 num_rows,
                                           int32 num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock);

/** This is analogous to the CUDA function hipGetErrorString(). **/
const char* cublasGetStatusStringK(hipblasStatus_t status);

/** This is analogous to the CUDA function hipGetErrorString(). **/
const char* cusparseGetStatusString(hipsparseStatus_t status);

/** This is analogous to the CUDA function hipGetErrorString(). **/
const char* curandGetStatusString(hiprandStatus_t status);

}  // namespace kaldi

#else  // HAVE CUDA
#define NVTX_RANGE(name)
#endif  // HAVE_CUDA

namespace kaldi {
// Some forward declarations, needed for friend declarations.
template<typename Real> class CuVectorBase;
template<typename Real> class CuVector;
template<typename Real> class CuSubVector;
template<typename Real> class CuRand;
template<typename Real> class CuMatrixBase;
template<typename Real> class CuMatrix;
template<typename Real> class CuSubMatrix;
template<typename Real> class CuPackedMatrix;
template<typename Real> class CuSpMatrix;
template<typename Real> class CuTpMatrix;
template<typename Real> class CuSparseMatrix;

template<typename Real> class CuBlockMatrix; // this has no non-CU counterpart.

}  // namespace kaldi

#endif  // KALDI_CUDAMATRIX_CU_COMMON_H_
