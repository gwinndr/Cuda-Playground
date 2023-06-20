#include "CudaUtils.hpp"
#include <cmath>
#include <iostream>

#include <Utils/CudaConstants.hpp>

void SetBlockAndGridX(dim3& arsGridSize, dim3& arsBlockSize, const int anNumElems)
{
	int lnNumWarps = std::ceil((float)anNumElems / (float)WARP_SIZE);
	int lnBlocks = std::ceil((float)lnNumWarps / (float)MAX_WARPS_PER_BLOCK);

	int lnWarpsPerBlock = std::ceil((float)lnNumWarps / (float)lnBlocks);

	arsGridSize.x = lnBlocks;
	arsBlockSize.x = lnWarpsPerBlock * WARP_SIZE;

	//std::cout << arsGridSize.x << " " << arsBlockSize.x << std::endl;
}
