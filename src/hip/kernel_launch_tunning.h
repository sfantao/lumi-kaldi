#undef hipLaunchKernelGGLInternal
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)  \
    do {                                                                                           \
        kernelName<<<(numBlocks), (numThreads), (memPerBlock), ( (streamId == 0) ? (hipStreamPerThread) : (streamId) )>>>(__VA_ARGS__);         \
    } while (0)
#else
#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)  \
    do {                                                                                           \
        kernelName<<<(numBlocks), (numThreads), (memPerBlock), ( (streamId == 0) ? (hipStreamDefault) : (streamId) )>>>(__VA_ARGS__);         \
    } while (0)
#endif
