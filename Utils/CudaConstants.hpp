const int WARP_SIZE = 32;
const int MAX_BLOCK_SIZE = 1024;

const int MAX_WARPS_PER_BLOCK = MAX_BLOCK_SIZE / WARP_SIZE;

const unsigned FULL_SHFL_MASK = 0xffffffff;


