ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef ROCMDIR
$(error ROCMDIR not defined.)
endif

CXXFLAGS += -DHAVE_CUDA=1 -D__IS_HIP_COMPILE__=1 -D__HIP_PLATFORM_AMD__=1 \
	    -I$(ROCMDIR)/include -I$(ROCMDIR)/hiprand/include -I$(ROCMDIR)/rocrand/include -fPIC -pthread -isystem $(OPENFSTINC)

ROCM_INCLUDE= -I$(ROCMDIR)/include -I$(ROCMDIR)/hiprand/include -I$(ROCMDIR)/rocrand/include -I.. -isystem $(OPENFSTINC)
ROCM_FLAGS = -E --offload-arch=gfx908 -fPIC -DHAVE_CUDA=1 \
             -D__IS_HIP_COMPILE__=1 -D__CUDACC_VER_MAJOR__=11 -D__CUDA_ARCH__=800 -DCUDA_VERSION=11000 \
	     -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) -std=c++14 -DCUDA_API_PER_THREAD_DEFAULT_STREAM

#CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64/stubs -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
#CUDA_LDFLAGS += -L$(CUDATKDIR)/lib/stubs -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib
ROCM_LDFLAGS += 

#CUDA_LDLIBS += -lcuda -lcublas -lcusparse -lcusolver -lcudart -lcurand -lcufft -lnvToolsExt
ROCM_LDLIBS += 
