#include <cuda_runtime.h> 
#include <cublas_v2.h>

#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x)

#define THREAD_NUM 1024
