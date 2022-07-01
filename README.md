[![Build Status](https://travis-ci.com/kaldi-asr/kaldi.svg?branch=master)](https://travis-ci.com/kaldi-asr/kaldi)
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/kaldi-asr/kaldi) 
Kaldi Speech Recognition Toolkit
================================

To build the toolkit: see `./INSTALL`.  These instructions are valid for UNIX
systems including various flavors of Linux; Darwin; and Cygwin (has not been
tested on more "exotic" varieties of UNIX).  For Windows installation
instructions (excluding Cygwin), see `windows/INSTALL`.

To run the example system builds, see `egs/README.txt`

If you encounter problems (and you probably will), please do not hesitate to
contact the developers (see below). In addition to specific questions, please
let us know if there are specific aspects of the project that you feel could be
improved, that you find confusing, etc., and which missing features you most
wish it had.

Kaldi information channels
--------------------------

For HOT news about Kaldi see [the project site](http://kaldi-asr.org/).

[Documentation of Kaldi](http://kaldi-asr.org/doc/):
- Info about the project, description of techniques, tutorial for C++ coding.
- Doxygen reference of the C++ code.

[Kaldi forums and mailing lists](http://kaldi-asr.org/forums.html):

We have two different lists
- User list kaldi-help
- Developer list kaldi-developers:

To sign up to any of those mailing lists, go to
[http://kaldi-asr.org/forums.html](http://kaldi-asr.org/forums.html):


Development pattern for contributors
------------------------------------

1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/)
   of the [main Kaldi repository](https://github.com/kaldi-asr/kaldi) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create
   a branch `my-awesome-feature`.
3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/)
   through the Web interface of GitHub.
4. As a general rule, please follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
   There are a [few exceptions in Kaldi](http://kaldi-asr.org/doc/style.html).
   You can use the [Google's cpplint.py](https://raw.githubusercontent.com/google/styleguide/gh-pages/cpplint/cpplint.py)
   to verify that your code is free of basic mistakes.

Platform specific notes
-----------------------

### PowerPC 64bits little-endian (ppc64le)

- Kaldi is expected to work out of the box in RHEL >= 7 and Ubuntu >= 16.04 with
  OpenBLAS, ATLAS, or CUDA.
- CUDA drivers for ppc64le can be found at [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- An [IBM Redbook](https://www.redbooks.ibm.com/abstracts/redp5169.html) is
  available as a guide to install and configure CUDA.

### Android

- Kaldi supports cross compiling for Android using Android NDK, clang++ and
  OpenBLAS.
- See [this blog post](http://jcsilva.github.io/2017/03/18/compile-kaldi-android/)
  for details.

### Web Assembly

- Kaldi supports cross compiling for Web Assembly for in-browser execution
  using [emscripten](https://emscripten.org/) and CLAPACK.
- See [this post](https://gitlab.inria.fr/kaldi.web/kaldi-wasm/-/wikis/build_details.md)
  for a step-by-step description of the build process.

## HIP build
Here's a script that can be used as reference to hipify and build the CUDA files:
```bash
#!/bin/bash

ml gcc/9.3.0 cmake patch cuda/11.3.0

set -ex

#
# Download Kaldi 
#
if [ 1 -eq 0 ] ; then
  rm -rf src
  git clone https://github.com/kaldi-asr/kaldi src
  # Internally it leaves at https://github.com/AMD-HPC/kaldi
fi

if [ 1 -eq 0 ] ; then
  rm -rf $(pwd)/miniconda3
  curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
  bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b p $(pwd)/miniconda3
fi

source miniconda3/bin/activate

if [ 1 -eq 0 ] ; then
  conda create -y -n kaldi python==2.7.18
fi

conda activate kaldi

if [ 1 -eq 0 ] ; then
  conda install -y mkl=2021.4.0=h06a4308_640 mkl-include=2021.4.0=h06a4308_640 svn=1.10.2=h52f66ed_0
  conda install -c conda-forge -y sox=14.4.2=h27f72bc_1013
fi

export MKL_ROOT=$(pwd)/miniconda3/envs/kaldi
export KALDI_CUDA_PATH=$(realpath $(dirname $(which ptxas))/../)

if [ 1 -eq 0 ] ; then
  cd src/tools
  ./extras/check_dependencies.sh  
  make -j |& tee sam_make.log
  cd -
fi

if [ 1 -eq 0 ] ; then
  git clone -b cuda-11.3 https://github.com/NVIDIA/cub src/tools/cub-git
fi

if [ 1 -eq 1 ] ; then
  cd src/tools
  rm -rf cub
  ln -s cub-git cub
  cd -
fi

if [ 1 -eq 0 ] ; then
  cd src/src
  
  ./configure \
    --shared \
    --debug-level=0 \
    --use-cuda --with-cudadecoder \
    --cudatk-dir="$KALDI_CUDA_PATH" \
    --cuda-arch="--gpu-architecture=compute_80 --gpu-code=sm_80" \
    --mkl-root="$MKL_ROOT"
    
  cd -
fi

if [ 1 -eq 0 ] ; then
  cd src/src
  make -j8 clean depend
  make EXTRA_CXXFLAGS="-O3 -DNDEBUG" EXTRA_LDFLAGS="" -j12 |& tee sam_make.log
  cd -
fi

#
# Hipify
#

cd src/src/cudadecoder
/share/modules/cuda/11.3.0/bin/nvcc -E cuda-decoder-kernels.cu -o cuda-decoder-kernels.i -I/share/modules/cuda/11.3.0/include -I/home/sfantao/lumi-builds/kaldi/src/tools/cub-git -I.. -isystem /home/sfantao/lumi-builds/kaldi/src/tools/openfst-1.7.2/include --compiler-options -fPIC --machine 64 -DHAVE_CUDA -ccbin c++ -DKALDI_DOUBLEPRECISION=0 -std=c++14 -DCUDA_API_PER_THREAD_DEFAULT_STREAM -lineinfo --verbose -Wno-deprecated-gpu-targets --gpu-architecture=compute_80 --gpu-code=sm_80 -I../
cd -  

ml rocm/5.0.0-9282

hp=""
hp="$hp src/src/chain/chain-kernels.cu"
hp="$hp src/src/cudadecoder/cuda-decoder-kernels.cu"
hp="$hp src/src/cudadecoder/batched-static-nnet3-kernels.cu"
hp="$hp src/src/cudafeat/feature-online-cmvn-cuda.cu"
hp="$hp src/src/cudafeat/feature-spectral-cuda.cu"
hp="$hp src/src/cudafeat/online-ivector-feature-cuda-kernels.cu"
hp="$hp src/src/cudafeat/feature-online-batched-spectral-cuda-kernels.cu"
hp="$hp src/src/cudafeat/feature-online-batched-cmvn-cuda-kernels.cu"
hp="$hp src/src/cudafeat/feature-window-cuda.cu"
hp="$hp src/src/cudafeat/feature-online-batched-ivector-cuda-kernels.cu"
hp="$hp src/src/cudamatrix/cu-kernels.cu"

hp="$hp src/src/cudadecoder/batched-static-nnet3-kernels.h"
hp="$hp src/src/cudadecoder/cuda-decoder-kernels.h"
hp="$hp src/src/cudadecoder/cuda-decoder-kernels-utils.h"
hp="$hp src/src/cudadecoder/cuda-decoder-common.h"
hp="$hp src/src/cudamatrix/cu-common.h"
hp="$hp src/src/cudamatrix/cu-device.h"
hp="$hp src/src/cudamatrix/cu-allocator.h"
hp="$hp src/src/cudamatrix/cu-matrix.h"
hp="$hp src/src/cudamatrix/cu-value.h"
hp="$hp src/src/cudamatrix/cu-array-inl.h"
hp="$hp src/src/cudamatrix/cu-kernels.h"
hp="$hp src/src/cudafeat/feature-spectral-cuda.h"
hp="$hp src/src/cudafeat/feature-window-cuda.h"

if [ 1 -eq 1 ] ; then
  for i in $hp ; do
    j=$i.hip
    echo $j
    
    if [ -f $j ] ; then
      continue
    fi
    
    # Hipify if we didn't do that already.
    /opt/rocm-5.0.0-9003/bin/hipify-perl -print-stats $i 1> $j 2>$j.stats
    
    # If we are hipifying a header file, direct the preprocessor to hipified version 
    # using the __IS_HIP_COMPILE__ macro to control it.
    if [[ $i == *.h ]] ; then
      mv $i $i.bak
      
cat >> $i << EOF
#ifdef __IS_HIP_COMPILE__
#include "$(basename $j)"
#else        
EOF

cat $i.bak >> $i

cat >> $i << EOF
#endif  //__IS_HIP_COMPILE__  
EOF

    fi
  done
fi


#
# Attempt build hipified versions
#

if [ 1 -eq 1 ] ; then
  for i in $(find src/src -iname *.cu.hip) ; do
    
    src=$(basename $i)
    dst=$src.o
    echo $dst
    
    if [ -f $i.o ]; then
      continue
    fi
    
    cd $(dirname $i)
    
    hipcc \
      -E \
      --offload-arch=gfx908 \
      -O3 \
      -D__IS_HIP_COMPILE__=1 \
      -D__CUDACC_VER_MAJOR__=11 \
      -D__CUDA_ARCH__=800 \
      -DCUDA_VERSION=11030 \
      -DHAVE_CUDA=1 \
      \
      -I.. -isystem /home/sfantao/lumi-builds/kaldi/src/tools/openfst-1.7.2/include \
      -fPIC \
      -DKALDI_DOUBLEPRECISION=0 \
      -std=c++14 \
      -DCUDA_API_PER_THREAD_DEFAULT_STREAM \
      \
      -o $dst.i $src  
      
    # /share/modules/cuda/11.3.0/bin/nvcc -E cuda-decoder-kernels.cu -o cuda-decoder-kernels.i -I/share/modules/cuda/11.3.0/include -I/home/sfantao/lumi-builds/kaldi/src/tools/cub-git -I.. -isystem /home/sfantao/lumi-builds/kaldi/src/tools/openfst-1.7.2/include --compiler-options -fPIC --machine 64 -DHAVE_CUDA -ccbin c++ -DKALDI_DOUBLEPRECISION=0 -std=c++14 -DCUDA_API_PER_THREAD_DEFAULT_STREAM -lineinfo --verbose -Wno-deprecated-gpu-targets --gpu-architecture=compute_80 --gpu-code=sm_80 -I../
    hipcc \
      -c \
      --offload-arch=gfx908 \
      -O3 \
      -D__IS_HIP_COMPILE__=1 \
      -D__CUDACC_VER_MAJOR__=11 \
      -D__CUDA_ARCH__=800 \
      -DCUDA_VERSION=11030 \
      -DHAVE_CUDA=1 \
      \
      -I.. -isystem /home/sfantao/lumi-builds/kaldi/src/tools/openfst-1.7.2/include \
      -fPIC \
      -DKALDI_DOUBLEPRECISION=0 \
      -std=c++14 \
      -DCUDA_API_PER_THREAD_DEFAULT_STREAM \
      \
      -o $dst $src  
    cd -
  done
fi
```
