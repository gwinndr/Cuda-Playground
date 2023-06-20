#include "TestShuffle.hpp"

#include <CudaBuffer/CudaBuffer.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/CudaUtils.hpp>

#include <Kernels/ShuffleReduce.cuh>

#include <iostream>

TestShuffle::TestShuffle()
{

}

TestShuffle::~TestShuffle()
{

}

void TestShuffle::Test()
{
	std::cout << "TestShuffle start" << std::endl;
	std::cout << std::endl;

	constexpr int lnLen = 7211156;
	float* lpfReduceHost = new float[lnLen];

	// Needed to determine output buffer size
	dim3 grid_size;
	dim3 block_size;
	SetBlockAndGridX(grid_size, block_size, lnLen);

	// Need intermediate reductions for each block in the grid
	const int len_result = grid_size.x;
	float* lpfResultHost = new float[len_result];

	// dynamic shared memory is not needed
	constexpr int shared_mem = 0;

	// Set first buffer only
	for (int i = 0; i < lnLen; ++i)
	{
		lpfReduceHost[i] = 2;
	}

	CudaBuffer<float> lcReduceBuf(lnLen, lpfReduceHost);
	CudaBuffer<float> lcResultBuf(len_result, lpfResultHost);

	cudaStream_t lhStream;
	cudaStreamCreate(&lhStream);

	// Copy first buffer to dev
	lcReduceBuf.CopyHostToDev(lhStream);

	CallShuffleReduce(grid_size, block_size, shared_mem, 
		lcReduceBuf.GetDevPtr(), lcResultBuf.GetDevPtr(), lnLen, lhStream);

	lcResultBuf.CopyDevToHost(lhStream);

	cudaStreamSynchronize(lhStream);

	//std::cout << "ReduceBuffer: " << ArrayToStr<float>(lpfReduceHost, lnLen) << std::endl;
	std::cout << "Result: " << *lpfResultHost << std::endl;
	std::cout << "Expected Result: " << (float)(lnLen * 2) << std::endl;
	//for (int i = 0; i < len_result; ++i)
	//{
	//	std::cout << "Result[" << i << "]: " << lpfResultHost[i] << std::endl;
	//}

	std::cout << std::endl;
	std::cout << "TestShuffle end" << std::endl;
}
