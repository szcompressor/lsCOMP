#include "lsCOMP_entry.h"
#include "lsCOMP_kernel.h"

// just for debugging, remember to delete later.
#include <stdio.h> 
// Define a macro for error checking
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
// Function to check CUDA errors
void check(cudaError_t result, const char *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, (unsigned int)result, cudaGetErrorString(result));
        // Exit if there is an error
        exit(result);
    }
}


void lsCOMP_compression_uint32_bsize64(uint32_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    size_t* d_cmpOffset;
    size_t* d_locOffset;
    int* d_flag;
    size_t glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    lsCOMP_compression_kernel_uint32_bsize64<<<gridSize, blockSize, sizeof(size_t)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(size_t), cudaMemcpyDeviceToHost);
    *cmpSize = glob_sync + blockNum;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void lsCOMP_decompression_uint32_bsize64(uint32_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    size_t* d_cmpOffset;
    size_t* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    lsCOMP_decompression_kernel_uint32_bsize64<<<gridSize, blockSize, sizeof(size_t)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void lsCOMP_compression_uint16_bsize64(uint16_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    size_t* d_cmpOffset;
    size_t* d_locOffset;
    int* d_flag;
    size_t glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    lsCOMP_compression_kernel_uint16_bsize64<<<gridSize, blockSize, sizeof(size_t)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(size_t), cudaMemcpyDeviceToHost);
    *cmpSize = glob_sync + blockNum;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void lsCOMP_decompression_uint16_bsize64(uint16_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    size_t* d_cmpOffset;
    size_t* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    lsCOMP_decompression_kernel_uint16_bsize64<<<gridSize, blockSize, sizeof(size_t)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}
