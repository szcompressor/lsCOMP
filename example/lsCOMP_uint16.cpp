#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <lsCOMP_entry.h>
#include <lsCOMP_timer.h>
#include <lsCOMP_utility.h>

int main(int argc, char* argv[])
{
    // Read input information.
    char oriFilePath[640] = {0};
    char cmpFilePath[640] = {0};
    char decFilePath[640] = {0};
    uint3 dims = {0, 0, 0};
    uint4 quantBins = {0, 0, 0, 0};
    float poolingTH = 0.0f;
    int status=0;

    // Check if enough arguments are provided
    if (argc < 13) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "   %s -i oriFilePath -d dims.x dims.y dims.z -b quantBins.x quantBins.y quantBins.z quantBins.w -p value -o decFilePath\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "   -i oriFilePath: Path to the original data file\n");
        fprintf(stderr, "   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.\n");
        fprintf(stderr, "   -b quantBins.x quantBins.y quantBins.z quantBins.w: Quantization bins for the 4 levels, where x is the base one and x<=y<=z<=w.\n");
        fprintf(stderr, "   -p value: Pooling threshold for a data block.\n");
        fprintf(stderr, "   -x cmpFilePath: Path to the compressed data file   (optional).\n");
        fprintf(stderr, "   -o decFilePath: Path to the decompressed data file (optional).\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "   %s -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5\n", argv[0]);
        fprintf(stderr, "   %s -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5 -x data/xpcs-cmp.bin\n", argv[0]);
        fprintf(stderr, "   %s -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5 -o data/xpcs-dec.bin\n", argv[0]);
        fprintf(stderr, "   %s -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5 -x data/xpcs-cmp.bin -o data/xpcs-dec.bin\n", argv[0]);
        return 1;
    }

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                strncpy(oriFilePath, argv[i + 1], sizeof(oriFilePath) - 1);
                i++;
            } else {
                fprintf(stderr, "Error: Missing value for -i\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0) {
            if (i + 3 < argc) {
                dims.x = atoi(argv[i + 1]);
                dims.y = atoi(argv[i + 2]);
                dims.z = atoi(argv[i + 3]);
                if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
                    fprintf(stderr, "Error: -d values must be positive integers\n");
                    return 1;
                }
                i += 3;
            } else {
                fprintf(stderr, "Error: Missing values for -d\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 4 < argc) {
                quantBins.x = atoi(argv[i + 1]);
                quantBins.y = atoi(argv[i + 2]);
                quantBins.z = atoi(argv[i + 3]);
                quantBins.w = atoi(argv[i + 4]);
                if (quantBins.x <= 0 || quantBins.y <= 0 || quantBins.z <= 0 || quantBins.w <= 0) {
                    fprintf(stderr, "Error: -b values must be positive integers\n");
                    return 1;
                }
                if (!(quantBins.x <= quantBins.y && quantBins.y <= quantBins.z && quantBins.z <= quantBins.w)) {
                    fprintf(stderr, "Error: -b values must satisfy x <= y <= z <= w\n");
                    return 1;
                }
                i += 4;
            } else {
                fprintf(stderr, "Error: Missing values for -b\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                poolingTH = atof(argv[i + 1]);
                if (poolingTH < 0.0f || poolingTH > 1.0f) {
                    fprintf(stderr, "Error: -p value must be between 0 and 1\n");
                    return 1;
                }
                i++;
            } else {
                fprintf(stderr, "Error: Missing value for -p\n");
                return 1;
            }            
        } else if (strcmp(argv[i], "-x") == 0) {
            if (i + 1 < argc) {
                strncpy(cmpFilePath, argv[i + 1], sizeof(cmpFilePath) - 1);
                i++;
            } else {
                fprintf(stderr, "Error: Missing value for -o\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                strncpy(decFilePath, argv[i + 1], sizeof(decFilePath) - 1);
                i++;
            } else {
                fprintf(stderr, "Error: Missing value for -o\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    // // Yafan is checking correctness of input values
    // printf("Parsed Values:\n");
    // printf("oriFilePath: %s\n", oriFilePath);
    // printf("cmpFilePath: %s\n", cmpFilePath);
    // printf("decFilePath: %s\n", decFilePath);
    // printf("dims: %u x %u x %u\n", dims.x, dims.y, dims.z);
    // printf("quantBins: %u %u %u %u\n", quantBins.x, quantBins.y, quantBins.z, quantBins.w);
    // printf("poolingTH: %.2f\n", poolingTH);

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on GPU.
    uint16_t* oriData = NULL;
    uint16_t* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    oriData = readUInt16Data_Yafan(oriFilePath, &nbEle, &status);
    if(nbEle != (size_t)dims.x * (size_t)dims.y * (size_t)dims.z) {
        fprintf(stderr, "Error: The number of elements in the original data does not match the dimensions\n");
        return 1;
    }
    decData = (uint16_t*)malloc(nbEle * sizeof(uint16_t));
    cmpBytes = (unsigned char*)malloc(nbEle * sizeof(uint16_t));
    
    // Input data preparation on GPU.
    uint16_t* d_oriData;
    uint16_t* d_decData;
    unsigned char* d_cmpBytes;
    cudaMalloc((void**)&d_oriData, nbEle*sizeof(uint16_t));
    cudaMemcpy(d_oriData, oriData, nbEle*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, nbEle*sizeof(uint16_t));
    cudaMalloc((void**)&d_cmpBytes, nbEle*sizeof(uint16_t));

    // Initialize CUDA stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup for NVIDIA GPU.
    for(int i=0; i<3; i++) lsCOMP_compression_uint16_bsize64(d_oriData, d_cmpBytes, &cmpSize, dims, quantBins, poolingTH, stream);
    printf("GPU warmup finished!\n\n");

    // lsCOMP compression.
    timer_GPU.StartCounter();
    lsCOMP_compression_uint16_bsize64(d_oriData, d_cmpBytes, &cmpSize, dims, quantBins, poolingTH, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, nbEle*sizeof(uint16_t)); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice); // copy back to GPU.

    // lsCOMP decompression.
    timer_GPU.StartCounter();
    lsCOMP_decompression_uint16_bsize64(d_decData, d_cmpBytes, cmpSize, dims, quantBins, poolingTH, stream);
    float decTime = timer_GPU.GetCounter();

    // Print results.
    printf("Dataset information:\n");
    printf("  - dims:       %u x %u x %u\n", dims.x, dims.y, dims.z);
    printf("  - length:     %zu\n", nbEle);
    printf("  - oriSize:    %f GB\n", nbEle*sizeof(uint16_t)/1024.0/1024.0/1024.0);
    printf("  - cmpSize:    %f GB\n\n", cmpSize*sizeof(unsigned char)/1024.0/1024.0/1024.0);
    printf("Input arguments:\n");
    printf("  - quantBins:  %u %u %u %u\n", quantBins.x, quantBins.y, quantBins.z, quantBins.w);
    printf("  - poolingTH:  %f\n\n", poolingTH);
    printf("lsCOMP compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(uint16_t)/1024.0/1024.0)/cmpTime);
    printf("lsCOMP decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(uint16_t)/1024.0/1024.0)/decTime);
    printf("lsCOMP compression ratio: %f\n", (nbEle*sizeof(uint16_t)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));

    // Write compressed data if needed.
    if(strlen(cmpFilePath) > 0) {
        cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        writeByteData_Yafan(cmpBytes, cmpSize, cmpFilePath, &status);
    }

    // Write reconstructed data if needed.
    if(strlen(decFilePath) > 0) {
        cudaMemcpy(decData, d_decData, nbEle*sizeof(uint16_t), cudaMemcpyDeviceToHost);
        writeUShortData_inBytes_Yafan(decData, nbEle, decFilePath, &status);
    }

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);
    return 0;
}