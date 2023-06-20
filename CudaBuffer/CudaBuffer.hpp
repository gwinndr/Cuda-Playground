#include <cuda_runtime.h>
#include <iostream>

template <class T>
class CudaBuffer
{
public:
	// Create buffer of specified length and gives ownership of the given host ptr
	//
	// NOTE: Host ptr must be of length anLen. This class will take ownership of this
	// ptr and handle freeing in the destructor
	CudaBuffer(int anLen, T* aphHostPtr=nullptr);

	~CudaBuffer();

	T* GetHostPtr()
	{
		return mphHostPtr;
	}

	T* GetDevPtr()
	{
		return mphDevPtr;
	}

	cudaError_t CopyHostToDev(cudaStream_t ahStream = 0);

	cudaError_t CopyDevToHost(cudaStream_t ahStream = 0);

	cudaError_t CopyToBuffer(CudaBuffer<T>& arcBuffer, cudaStream_t ahStream = 0);

private:

	void ThrowIfPtrsBad(bool abCheckHost = true, bool abCheckDev = true);

	int mnLen = 0;
	int mnNumBytes = 0;

	T* mphHostPtr = nullptr;
	T* mphDevPtr = nullptr;
	
};


template <class T>
CudaBuffer<T>::CudaBuffer(int anLen, T* aphHostPtr)
{
	if (aphHostPtr)
	{
		mphHostPtr = aphHostPtr;
	}

	mnLen = anLen;
	mnNumBytes = anLen * sizeof(T);

	cudaMalloc(&mphDevPtr, mnNumBytes);
}

template <class T>
CudaBuffer<T>::~CudaBuffer()
{
	if (mphHostPtr)
	{
		//std::cout << "Cleaning host ptr" << std::endl;
		delete[] mphHostPtr;
		mphHostPtr = nullptr;
	}

	if (mphDevPtr)
	{
		//std::cout << "Cleaning dev ptr" << std::endl;
		cudaFree(mphDevPtr);
		mphDevPtr = nullptr;
	}
}

template <class T>
cudaError_t CudaBuffer<T>::CopyHostToDev(cudaStream_t ahStream)
{
	ThrowIfPtrsBad();
	return cudaMemcpyAsync(mphDevPtr, mphHostPtr, mnNumBytes,
		cudaMemcpyHostToDevice, ahStream);
}

template <class T>
cudaError_t CudaBuffer<T>::CopyDevToHost(cudaStream_t ahStream)
{
	ThrowIfPtrsBad();
	return cudaMemcpyAsync(mphHostPtr, mphDevPtr, mnNumBytes,
		cudaMemcpyDeviceToHost, ahStream);
}

template <class T>
cudaError_t CudaBuffer<T>::CopyToBuffer(CudaBuffer<T>& arcBuffer, cudaStream_t ahStream)
{
	ThrowIfPtrsBad(false, true);
	arcBuffer.ThrowIfPtrsBad(false, true);

	return cudaMemcpyAsync(arcBuffer.mphDevPtr, mphDevPtr, mnNumBytes,
		cudaMemcpyDeviceToDevice, ahStream);
}

template <class T>
void CudaBuffer<T>::ThrowIfPtrsBad(bool abCheckHost, bool abCheckDev)
{
	if (abCheckHost && mphHostPtr == nullptr)
	{
		throw std::runtime_error("Host ptr is nullptr");
	}

	if (abCheckDev && mphDevPtr == nullptr)
	{
		throw std::runtime_error("Dev ptr is nullptr");
	}
}


