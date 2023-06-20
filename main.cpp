#include <iostream>

#include <TestCode/TestCudaBuffer.hpp>
#include <TestCode/TestShuffle.hpp>

int main()
{
	//std::cout << "Main" << std::endl;

	// Run code you want from here

	// ======== TEST CUDA BUFFER ======== //
	/*TestCudaBuffer lcTestCudaBuffer;
	lcTestCudaBuffer.Test();*/

	// ======== TEST CUDA BUFFER ======== //
	TestShuffle lcTestShuffle;
	lcTestShuffle.Test();

	return 0;
}


