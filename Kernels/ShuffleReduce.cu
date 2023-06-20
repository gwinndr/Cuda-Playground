#include "ShuffleReduce.cuh"
#include <cuda_runtime.h>

#include <Utils/CudaConstants.hpp>
#include <Utils/CudaUtils.hpp>

__device__
float ShuffleReduceWarp(float val)
{
	val += __shfl_xor_sync(FULL_SHFL_MASK, val, 16);
	val += __shfl_xor_sync(FULL_SHFL_MASK, val, 8);
	val += __shfl_xor_sync(FULL_SHFL_MASK, val, 4);
	val += __shfl_xor_sync(FULL_SHFL_MASK, val, 2);
	val += __shfl_xor_sync(FULL_SHFL_MASK, val, 1);

	return val;
}


__device__
void ShuffleReduceBlock(const float* A, float* B, const int anLenA)
{
	static __shared__ float shared[32];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int laneid = threadIdx.x % WARP_SIZE;
	int warp_id = threadIdx.x / WARP_SIZE;

	float elem = 0.0;
	if (idx < anLenA)
	{
		elem = A[idx];
	}

	elem = ShuffleReduceWarp(elem);
	if (laneid == 0)
	{
		shared[warp_id] = elem;
	}
	__syncthreads();


	if (warp_id == 0)
	{
		elem = 0.0;
		if (laneid * WARP_SIZE < anLenA)
		{
			elem = shared[laneid];
		}
		
		elem = ShuffleReduceWarp(elem);

		if(laneid == 0)
			*B = elem;
	}
}

__global__
void ShuffleReduce(const float* A, float* B, const int anLenA)
{
	// Each block will reduce
	float* out_addr = B + blockIdx.x;
	ShuffleReduceBlock(A, out_addr, anLenA);
}

void CallShuffleReduce(
	dim3 grid_size, dim3 block_size, int shared_mem,
	const float* A, float* B, const int anLenA, cudaStream_t stream)
{
	
	const float* cur_A = A;
	int cur_lenA = anLenA;
	while (cur_lenA > 1)
	{
		ShuffleReduce << <grid_size, block_size, shared_mem, stream >> > (
			cur_A, B, cur_lenA);

		// Set for next iteration to process intermediate reductions
		cur_A = B;
		cur_lenA = grid_size.x;

		SetBlockAndGridX(grid_size, block_size, cur_lenA);
	}
}