#ifndef LSCOMP_INCLUDE_LSCOMP_ENTRY_H
#define LSCOMP_INCLUDE_LSCOMP_ENTRY_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

void lsCOMP_compression_uint32_bsize64(uint32_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void lsCOMP_decompression_uint32_bsize64(uint32_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void lsCOMP_compression_uint16_bsize64(uint16_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void lsCOMP_decompression_uint16_bsize64(uint16_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);

#endif // LSCOMP_INCLUDE_LSCOMP_ENTRY_H