#ifndef LSCOMP_INCLUDE_LSCOMP_KERNEL_H
#define LSCOMP_INCLUDE_LSCOMP_KERNEL_H

#include <stdint.h>
#include <cuda_runtime.h>

static const int block_per_thread = 32;

__global__ void lsCOMP_compression_kernel_uint32_bsize64(const uint32_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);
__global__ void lsCOMP_decompression_kernel_uint32_bsize64(uint32_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);
__global__ void lsCOMP_compression_kernel_uint16_bsize64(const uint16_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);
__global__ void lsCOMP_decompression_kernel_uint16_bsize64(uint16_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);


#endif // LSCOMP_INCLUDE_LSCOMP_KERNEL_H