#include <cuda_runtime.h>

void CallShuffleReduce(dim3 grid_size, dim3 block_size, int shared_mem,
	const float* A, float* B, const int anLenA, cudaStream_t stream = 0);


