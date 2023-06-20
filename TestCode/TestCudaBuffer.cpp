#include "TestCudaBuffer.hpp"
#include <Utils/StringUtils.hpp>

TestCudaBuffer::TestCudaBuffer()
{

}

TestCudaBuffer::~TestCudaBuffer()
{

}


void TestCudaBuffer::Test()
{
	std::cout << "TestCudaBuffer start" << std::endl;
	std::cout << std::endl;

	constexpr int lnLen = 10;
	float* lpfHost1 = new float[lnLen];
	float* lpfHost3 = new float[lnLen];

	// Set first buffer only
	for (int i = 0; i < lnLen; ++i)
	{
		lpfHost1[i] = i + 1;
	}

	CudaBuffer<float> lcBuf1(lnLen, lpfHost1);
	CudaBuffer<float> lcBuf2(lnLen);
	CudaBuffer<float> lcBuf3(lnLen, lpfHost3);

	cudaStream_t lhStream;
	cudaStreamCreate(&lhStream);

	// Copy first buffer to dev
	lcBuf1.CopyHostToDev(lhStream);

	// Copy dev to dev to second buffer
	lcBuf1.CopyToBuffer(lcBuf2, lhStream);

	// Copy dev to dev to third buffer
	lcBuf2.CopyToBuffer(lcBuf3, lhStream);

	// Copy dev to host for the third buffer
	lcBuf3.CopyDevToHost(lhStream);

	// Print results
	std::cout << "Host1: " << ArrayToStr<float>(lpfHost1, lnLen) << std::endl;
	std::cout << std::endl;
	std::cout << "Host3: " << ArrayToStr<float>(lpfHost3, lnLen) << std::endl;

	std::cout << std::endl;
	std::cout << "TestCudaBuffer end" << std::endl;
}
