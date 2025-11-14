#include "lsCOMP_kernel.h"
#include <stdio.h> // just for debugging, remember to delete later.

__global__ void lsCOMP_compression_kernel_uint32_bsize64(const uint32_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ size_t excl_sum;
    __shared__ size_t base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    if(!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    size_t thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.
    
    // Scalar-quantization, Dynamic Binning Selection, Fixed-length Encoding.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }
            
            // Preparation for ratio profiling.
            uint zero_count = 0;
            uint zero_count_bins[4] = {0, 0, 0, 0};
            uint max_val1 = 0;
            uint max_val2 = 0;
            for(int i=0; i<64; i++)
            {
                uint val = block_data[i];
                zero_count += (val == 0);
                zero_count_bins[0] += (val < quant_bins[0]); // Base bin operation
                zero_count_bins[1] += (val < quant_bins[1]);
                zero_count_bins[2] += (val < quant_bins[2]);
                zero_count_bins[3] += (val < quant_bins[3]);
                max_val1 = (val > max_val1) ? val : max_val1;
                if(i%2)
                {
                    uint tmp_val = (block_data[i-1] + block_data[i]) / 2;
                    max_val2 = (tmp_val > max_val2) ? tmp_val : max_val2;
                }
            }

            // Compression algorithm selection and store meta data.
            float sparsity = (float)zero_count / 64;
            int pooling_choice = (sparsity > poolingTH);
            uint bin_choice = 0;
            // Progressively bin size selection.
            if(zero_count_bins[1]==zero_count_bins[0])
            {
                bin_choice = 1;
                if(zero_count_bins[2]==zero_count_bins[1])
                {
                    bin_choice = 2;
                    if(zero_count_bins[3]==zero_count_bins[2])
                    {
                        bin_choice = 3;
                    }
                }
            }

            // Store meta data.
            int max_quantized_val;
            int temp_rate = 0;
            if(pooling_choice)
            {
                max_quantized_val = max_val2 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 4;
                temp_rate = 0x80 | (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
            else
            {
                max_quantized_val = max_val1 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 8;
                temp_rate = (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            size_t loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also storing data to global memory.
    size_t base_cmp_byte_ofs = base_idx;
    size_t cmp_byte_ofs;
    size_t tmp_byte_ofs = 0;
    size_t cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }

            // Retrieve meta data.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;
            
            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            size_t prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                if(pooling_choice)
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) 
                    {
                        pooling_block_data[i] = (block_data[i*2] + block_data[i*2+1]) / 2;
                        pooling_block_data[i] = pooling_block_data[i] / quant_bins[bin_choice];
                    }

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer.x = 0;
                        tmp_buffer.y = 0;
                        tmp_buffer.z = 0;
                        tmp_buffer.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer.x = (((pooling_block_data[0] & mask) >> i) << 7) |
                                       (((pooling_block_data[1] & mask) >> i) << 6) |
                                       (((pooling_block_data[2] & mask) >> i) << 5) |
                                       (((pooling_block_data[3] & mask) >> i) << 4) |
                                       (((pooling_block_data[4] & mask) >> i) << 3) |
                                       (((pooling_block_data[5] & mask) >> i) << 2) |
                                       (((pooling_block_data[6] & mask) >> i) << 1) |
                                       (((pooling_block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer.y = (((pooling_block_data[8] & mask) >> i) << 7) |
                                       (((pooling_block_data[9] & mask) >> i) << 6) |
                                       (((pooling_block_data[10] & mask) >> i) << 5) |
                                       (((pooling_block_data[11] & mask) >> i) << 4) |
                                       (((pooling_block_data[12] & mask) >> i) << 3) |
                                       (((pooling_block_data[13] & mask) >> i) << 2) |
                                       (((pooling_block_data[14] & mask) >> i) << 1) |
                                       (((pooling_block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer.z = (((pooling_block_data[16] & mask) >> i) << 7) |
                                       (((pooling_block_data[17] & mask) >> i) << 6) |
                                       (((pooling_block_data[18] & mask) >> i) << 5) |
                                       (((pooling_block_data[19] & mask) >> i) << 4) |
                                       (((pooling_block_data[20] & mask) >> i) << 3) |
                                       (((pooling_block_data[21] & mask) >> i) << 2) |
                                       (((pooling_block_data[22] & mask) >> i) << 1) |
                                       (((pooling_block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer.w = (((pooling_block_data[24] & mask) >> i) << 7) |
                                       (((pooling_block_data[25] & mask) >> i) << 6) |
                                       (((pooling_block_data[26] & mask) >> i) << 5) |
                                       (((pooling_block_data[27] & mask) >> i) << 4) |
                                       (((pooling_block_data[28] & mask) >> i) << 3) |
                                       (((pooling_block_data[29] & mask) >> i) << 2) |
                                       (((pooling_block_data[30] & mask) >> i) << 1) |
                                       (((pooling_block_data[31] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer;
                        cmp_byte_ofs += 4;
                        mask <<= 1;  
                    }
                }
                else
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] / quant_bins[bin_choice];

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer1.x = 0;
                        tmp_buffer1.y = 0;
                        tmp_buffer1.z = 0;
                        tmp_buffer1.w = 0;
                        tmp_buffer2.x = 0;
                        tmp_buffer2.y = 0;
                        tmp_buffer2.z = 0;
                        tmp_buffer2.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer1.x = (((block_data[0] & mask) >> i) << 7) |
                                        (((block_data[1] & mask) >> i) << 6) |
                                        (((block_data[2] & mask) >> i) << 5) |
                                        (((block_data[3] & mask) >> i) << 4) |
                                        (((block_data[4] & mask) >> i) << 3) |
                                        (((block_data[5] & mask) >> i) << 2) |
                                        (((block_data[6] & mask) >> i) << 1) |
                                        (((block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer1.y = (((block_data[8] & mask) >> i) << 7) |
                                        (((block_data[9] & mask) >> i) << 6) |
                                        (((block_data[10] & mask) >> i) << 5) |
                                        (((block_data[11] & mask) >> i) << 4) |
                                        (((block_data[12] & mask) >> i) << 3) |
                                        (((block_data[13] & mask) >> i) << 2) |
                                        (((block_data[14] & mask) >> i) << 1) |
                                        (((block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer1.z = (((block_data[16] & mask) >> i) << 7) |
                                        (((block_data[17] & mask) >> i) << 6) |
                                        (((block_data[18] & mask) >> i) << 5) |
                                        (((block_data[19] & mask) >> i) << 4) |
                                        (((block_data[20] & mask) >> i) << 3) |
                                        (((block_data[21] & mask) >> i) << 2) |
                                        (((block_data[22] & mask) >> i) << 1) |
                                        (((block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer1.w = (((block_data[24] & mask) >> i) << 7) |
                                        (((block_data[25] & mask) >> i) << 6) |
                                        (((block_data[26] & mask) >> i) << 5) |
                                        (((block_data[27] & mask) >> i) << 4) |
                                        (((block_data[28] & mask) >> i) << 3) |
                                        (((block_data[29] & mask) >> i) << 2) |
                                        (((block_data[30] & mask) >> i) << 1) |
                                        (((block_data[31] & mask) >> i) << 0); 
                        
                        // Get i-th bit in 32~39 data.
                        tmp_buffer2.x = (((block_data[32] & mask) >> i) << 7) |
                                        (((block_data[33] & mask) >> i) << 6) |
                                        (((block_data[34] & mask) >> i) << 5) |
                                        (((block_data[35] & mask) >> i) << 4) |
                                        (((block_data[36] & mask) >> i) << 3) |
                                        (((block_data[37] & mask) >> i) << 2) |
                                        (((block_data[38] & mask) >> i) << 1) |
                                        (((block_data[39] & mask) >> i) << 0);
                        
                        // Get i-th bit in 40~47 data.
                        tmp_buffer2.y = (((block_data[40] & mask) >> i) << 7) |
                                        (((block_data[41] & mask) >> i) << 6) |
                                        (((block_data[42] & mask) >> i) << 5) |
                                        (((block_data[43] & mask) >> i) << 4) |
                                        (((block_data[44] & mask) >> i) << 3) |
                                        (((block_data[45] & mask) >> i) << 2) |
                                        (((block_data[46] & mask) >> i) << 1) |
                                        (((block_data[47] & mask) >> i) << 0);

                        // Get i-th bit in 48~55 data.
                        tmp_buffer2.z = (((block_data[48] & mask) >> i) << 7) |
                                        (((block_data[49] & mask) >> i) << 6) |
                                        (((block_data[50] & mask) >> i) << 5) |
                                        (((block_data[51] & mask) >> i) << 4) |
                                        (((block_data[52] & mask) >> i) << 3) |
                                        (((block_data[53] & mask) >> i) << 2) |
                                        (((block_data[54] & mask) >> i) << 1) |
                                        (((block_data[55] & mask) >> i) << 0);

                        // Get i-th bit in 56~63 data.
                        tmp_buffer2.w = (((block_data[56] & mask) >> i) << 7) |
                                        (((block_data[57] & mask) >> i) << 6) |
                                        (((block_data[58] & mask) >> i) << 5) |
                                        (((block_data[59] & mask) >> i) << 4) |
                                        (((block_data[60] & mask) >> i) << 3) |
                                        (((block_data[61] & mask) >> i) << 2) |
                                        (((block_data[62] & mask) >> i) << 1) |
                                        (((block_data[63] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer1;
                        cmp_byte_ofs += 4;
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer2;
                        cmp_byte_ofs += 4;
                        mask <<= 1; 
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}


__global__ void lsCOMP_decompression_kernel_uint32_bsize64(uint32_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ size_t excl_sum;
    __shared__ size_t base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    if(!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    size_t thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.

    // Obtain fixed-rate information for each block.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Obtain block meta data.
            fixed_rate[j] = cmpBytes[block_idx];

            // Check if pooling.
            int pooling_choice = fixed_rate[j] >> 7;
            int temp_rate = fixed_rate[j] & 0x1f;
            if(pooling_choice) thread_ofs += temp_rate * 4;
            else thread_ofs += temp_rate * 8;
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            size_t loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also reading data from global memory.
    size_t base_cmp_byte_ofs = base_idx;
    size_t cmp_byte_ofs;
    size_t tmp_byte_ofs = 0;
    size_t cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;
    
        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Initialization, guiding decoding process.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;

            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            size_t prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                // Buffering decompressed block data.
                uint block_data[64];

                // Read data and shuffle it back from global memory.
                if(pooling_choice)
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) pooling_block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        pooling_block_data[0] |= ((tmp_buffer.x >> 7) & 0x00000001) << i;
                        pooling_block_data[1] |= ((tmp_buffer.x >> 6) & 0x00000001) << i;
                        pooling_block_data[2] |= ((tmp_buffer.x >> 5) & 0x00000001) << i;
                        pooling_block_data[3] |= ((tmp_buffer.x >> 4) & 0x00000001) << i;
                        pooling_block_data[4] |= ((tmp_buffer.x >> 3) & 0x00000001) << i;
                        pooling_block_data[5] |= ((tmp_buffer.x >> 2) & 0x00000001) << i;
                        pooling_block_data[6] |= ((tmp_buffer.x >> 1) & 0x00000001) << i;
                        pooling_block_data[7] |= ((tmp_buffer.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        pooling_block_data[8] |= ((tmp_buffer.y >> 7) & 0x00000001) << i;
                        pooling_block_data[9] |= ((tmp_buffer.y >> 6) & 0x00000001) << i;
                        pooling_block_data[10] |= ((tmp_buffer.y >> 5) & 0x00000001) << i;
                        pooling_block_data[11] |= ((tmp_buffer.y >> 4) & 0x00000001) << i;
                        pooling_block_data[12] |= ((tmp_buffer.y >> 3) & 0x00000001) << i;
                        pooling_block_data[13] |= ((tmp_buffer.y >> 2) & 0x00000001) << i;
                        pooling_block_data[14] |= ((tmp_buffer.y >> 1) & 0x00000001) << i;
                        pooling_block_data[15] |= ((tmp_buffer.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        pooling_block_data[16] |= ((tmp_buffer.z >> 7) & 0x00000001) << i;
                        pooling_block_data[17] |= ((tmp_buffer.z >> 6) & 0x00000001) << i;
                        pooling_block_data[18] |= ((tmp_buffer.z >> 5) & 0x00000001) << i;
                        pooling_block_data[19] |= ((tmp_buffer.z >> 4) & 0x00000001) << i;
                        pooling_block_data[20] |= ((tmp_buffer.z >> 3) & 0x00000001) << i;
                        pooling_block_data[21] |= ((tmp_buffer.z >> 2) & 0x00000001) << i;
                        pooling_block_data[22] |= ((tmp_buffer.z >> 1) & 0x00000001) << i;
                        pooling_block_data[23] |= ((tmp_buffer.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        pooling_block_data[24] |= ((tmp_buffer.w >> 7) & 0x00000001) << i;
                        pooling_block_data[25] |= ((tmp_buffer.w >> 6) & 0x00000001) << i;
                        pooling_block_data[26] |= ((tmp_buffer.w >> 5) & 0x00000001) << i;
                        pooling_block_data[27] |= ((tmp_buffer.w >> 4) & 0x00000001) << i;
                        pooling_block_data[28] |= ((tmp_buffer.w >> 3) & 0x00000001) << i;
                        pooling_block_data[29] |= ((tmp_buffer.w >> 2) & 0x00000001) << i;
                        pooling_block_data[30] |= ((tmp_buffer.w >> 1) & 0x00000001) << i;
                        pooling_block_data[31] |= ((tmp_buffer.w >> 0) & 0x00000001) << i;
                    }

                    // Assign data back to block data.
                    for(int i=0; i<32; i++)
                    {
                        block_data[i*2] = pooling_block_data[i] * quant_bins[bin_choice];
                        block_data[i*2+1] = block_data[i*2];
                    }
                }
                else
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer1 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;
                        tmp_buffer2 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        block_data[0] |= ((tmp_buffer1.x >> 7) & 0x00000001) << i;
                        block_data[1] |= ((tmp_buffer1.x >> 6) & 0x00000001) << i;
                        block_data[2] |= ((tmp_buffer1.x >> 5) & 0x00000001) << i;
                        block_data[3] |= ((tmp_buffer1.x >> 4) & 0x00000001) << i;
                        block_data[4] |= ((tmp_buffer1.x >> 3) & 0x00000001) << i;
                        block_data[5] |= ((tmp_buffer1.x >> 2) & 0x00000001) << i;
                        block_data[6] |= ((tmp_buffer1.x >> 1) & 0x00000001) << i;
                        block_data[7] |= ((tmp_buffer1.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        block_data[8] |= ((tmp_buffer1.y >> 7) & 0x00000001) << i;
                        block_data[9] |= ((tmp_buffer1.y >> 6) & 0x00000001) << i;
                        block_data[10] |= ((tmp_buffer1.y >> 5) & 0x00000001) << i;
                        block_data[11] |= ((tmp_buffer1.y >> 4) & 0x00000001) << i;
                        block_data[12] |= ((tmp_buffer1.y >> 3) & 0x00000001) << i;
                        block_data[13] |= ((tmp_buffer1.y >> 2) & 0x00000001) << i;
                        block_data[14] |= ((tmp_buffer1.y >> 1) & 0x00000001) << i;
                        block_data[15] |= ((tmp_buffer1.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        block_data[16] |= ((tmp_buffer1.z >> 7) & 0x00000001) << i;
                        block_data[17] |= ((tmp_buffer1.z >> 6) & 0x00000001) << i;
                        block_data[18] |= ((tmp_buffer1.z >> 5) & 0x00000001) << i;
                        block_data[19] |= ((tmp_buffer1.z >> 4) & 0x00000001) << i;
                        block_data[20] |= ((tmp_buffer1.z >> 3) & 0x00000001) << i;
                        block_data[21] |= ((tmp_buffer1.z >> 2) & 0x00000001) << i;
                        block_data[22] |= ((tmp_buffer1.z >> 1) & 0x00000001) << i;
                        block_data[23] |= ((tmp_buffer1.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        block_data[24] |= ((tmp_buffer1.w >> 7) & 0x00000001) << i;
                        block_data[25] |= ((tmp_buffer1.w >> 6) & 0x00000001) << i;
                        block_data[26] |= ((tmp_buffer1.w >> 5) & 0x00000001) << i;
                        block_data[27] |= ((tmp_buffer1.w >> 4) & 0x00000001) << i;
                        block_data[28] |= ((tmp_buffer1.w >> 3) & 0x00000001) << i;
                        block_data[29] |= ((tmp_buffer1.w >> 2) & 0x00000001) << i;
                        block_data[30] |= ((tmp_buffer1.w >> 1) & 0x00000001) << i;
                        block_data[31] |= ((tmp_buffer1.w >> 0) & 0x00000001) << i;

                        // Get ith bit in 32~39 abs quant from global memory.
                        block_data[32] |= ((tmp_buffer2.x >> 7) & 0x00000001) << i;
                        block_data[33] |= ((tmp_buffer2.x >> 6) & 0x00000001) << i;
                        block_data[34] |= ((tmp_buffer2.x >> 5) & 0x00000001) << i;
                        block_data[35] |= ((tmp_buffer2.x >> 4) & 0x00000001) << i;
                        block_data[36] |= ((tmp_buffer2.x >> 3) & 0x00000001) << i;
                        block_data[37] |= ((tmp_buffer2.x >> 2) & 0x00000001) << i;
                        block_data[38] |= ((tmp_buffer2.x >> 1) & 0x00000001) << i;
                        block_data[39] |= ((tmp_buffer2.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 40~47 abs quant from global memory.
                        block_data[40] |= ((tmp_buffer2.y >> 7) & 0x00000001) << i;
                        block_data[41] |= ((tmp_buffer2.y >> 6) & 0x00000001) << i;
                        block_data[42] |= ((tmp_buffer2.y >> 5) & 0x00000001) << i;
                        block_data[43] |= ((tmp_buffer2.y >> 4) & 0x00000001) << i;
                        block_data[44] |= ((tmp_buffer2.y >> 3) & 0x00000001) << i;
                        block_data[45] |= ((tmp_buffer2.y >> 2) & 0x00000001) << i;
                        block_data[46] |= ((tmp_buffer2.y >> 1) & 0x00000001) << i;
                        block_data[47] |= ((tmp_buffer2.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 48-55 abs quant from global memory.
                        block_data[48] |= ((tmp_buffer2.z >> 7) & 0x00000001) << i;
                        block_data[49] |= ((tmp_buffer2.z >> 6) & 0x00000001) << i;
                        block_data[50] |= ((tmp_buffer2.z >> 5) & 0x00000001) << i;
                        block_data[51] |= ((tmp_buffer2.z >> 4) & 0x00000001) << i;
                        block_data[52] |= ((tmp_buffer2.z >> 3) & 0x00000001) << i;
                        block_data[53] |= ((tmp_buffer2.z >> 2) & 0x00000001) << i;
                        block_data[54] |= ((tmp_buffer2.z >> 1) & 0x00000001) << i;
                        block_data[55] |= ((tmp_buffer2.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 56-63 abs quant from global memory.
                        block_data[56] |= ((tmp_buffer2.w >> 7) & 0x00000001) << i;
                        block_data[57] |= ((tmp_buffer2.w >> 6) & 0x00000001) << i;
                        block_data[58] |= ((tmp_buffer2.w >> 5) & 0x00000001) << i;
                        block_data[59] |= ((tmp_buffer2.w >> 4) & 0x00000001) << i;
                        block_data[60] |= ((tmp_buffer2.w >> 3) & 0x00000001) << i;
                        block_data[61] |= ((tmp_buffer2.w >> 2) & 0x00000001) << i;
                        block_data[62] |= ((tmp_buffer2.w >> 1) & 0x00000001) << i;
                        block_data[63] |= ((tmp_buffer2.w >> 0) & 0x00000001) << i;
                    }

                    // Restore quantized data.
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] * quant_bins[bin_choice];
                }

                // Write data back to global memory.
                data_idx_x = block_idx_x;
                for(uint i=0; i<8; i++)
                {
                    data_idx_y = block_idx_y * 8 + i;
                    for(uint k=0; k<8; k++)
                    {
                        data_idx_z = block_idx_z * 8 + k;
                        data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                        if(data_idx_y < dims.y && data_idx_z < dims.z) decData[data_idx] = block_data[i*8+k];
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}


__global__ void lsCOMP_compression_kernel_uint16_bsize64(const uint16_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ size_t excl_sum;
    __shared__ size_t base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    if(!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    size_t thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.
    
    // Scalar-quantization, Dynamic Binning Selection, Fixed-length Encoding.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }
            
            // Preparation for ratio profiling.
            uint zero_count = 0;
            uint zero_count_bins[4] = {0, 0, 0, 0};
            uint max_val1 = 0;
            uint max_val2 = 0;
            for(int i=0; i<64; i++)
            {
                uint val = block_data[i];
                zero_count += (val == 0);
                zero_count_bins[0] += (val < quant_bins[0]); // Base bin operation
                zero_count_bins[1] += (val < quant_bins[1]);
                zero_count_bins[2] += (val < quant_bins[2]);
                zero_count_bins[3] += (val < quant_bins[3]);
                max_val1 = (val > max_val1) ? val : max_val1;
                if(i%2)
                {
                    uint tmp_val = (block_data[i-1] + block_data[i]) / 2;
                    max_val2 = (tmp_val > max_val2) ? tmp_val : max_val2;
                }
            }

            // Compression algorithm selection and store meta data.
            float sparsity = (float)zero_count / 64;
            int pooling_choice = (sparsity > poolingTH);
            uint bin_choice = 0;
            // Progressively bin size selection.
            if(zero_count_bins[1]==zero_count_bins[0])
            {
                bin_choice = 1;
                if(zero_count_bins[2]==zero_count_bins[1])
                {
                    bin_choice = 2;
                    if(zero_count_bins[3]==zero_count_bins[2])
                    {
                        bin_choice = 3;
                    }
                }
            }

            // Store meta data.
            int max_quantized_val;
            int temp_rate = 0;
            if(pooling_choice)
            {
                max_quantized_val = max_val2 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 4;
                temp_rate = 0x80 | (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
            else
            {
                max_quantized_val = max_val1 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 8;
                temp_rate = (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            size_t loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also storing data to global memory.
    size_t base_cmp_byte_ofs = base_idx;
    size_t cmp_byte_ofs;
    size_t tmp_byte_ofs = 0;
    size_t cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }

            // Retrieve meta data.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;
            
            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            size_t prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                if(pooling_choice)
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) 
                    {
                        pooling_block_data[i] = (block_data[i*2] + block_data[i*2+1]) / 2;
                        pooling_block_data[i] = pooling_block_data[i] / quant_bins[bin_choice];
                    }

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer.x = 0;
                        tmp_buffer.y = 0;
                        tmp_buffer.z = 0;
                        tmp_buffer.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer.x = (((pooling_block_data[0] & mask) >> i) << 7) |
                                       (((pooling_block_data[1] & mask) >> i) << 6) |
                                       (((pooling_block_data[2] & mask) >> i) << 5) |
                                       (((pooling_block_data[3] & mask) >> i) << 4) |
                                       (((pooling_block_data[4] & mask) >> i) << 3) |
                                       (((pooling_block_data[5] & mask) >> i) << 2) |
                                       (((pooling_block_data[6] & mask) >> i) << 1) |
                                       (((pooling_block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer.y = (((pooling_block_data[8] & mask) >> i) << 7) |
                                       (((pooling_block_data[9] & mask) >> i) << 6) |
                                       (((pooling_block_data[10] & mask) >> i) << 5) |
                                       (((pooling_block_data[11] & mask) >> i) << 4) |
                                       (((pooling_block_data[12] & mask) >> i) << 3) |
                                       (((pooling_block_data[13] & mask) >> i) << 2) |
                                       (((pooling_block_data[14] & mask) >> i) << 1) |
                                       (((pooling_block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer.z = (((pooling_block_data[16] & mask) >> i) << 7) |
                                       (((pooling_block_data[17] & mask) >> i) << 6) |
                                       (((pooling_block_data[18] & mask) >> i) << 5) |
                                       (((pooling_block_data[19] & mask) >> i) << 4) |
                                       (((pooling_block_data[20] & mask) >> i) << 3) |
                                       (((pooling_block_data[21] & mask) >> i) << 2) |
                                       (((pooling_block_data[22] & mask) >> i) << 1) |
                                       (((pooling_block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer.w = (((pooling_block_data[24] & mask) >> i) << 7) |
                                       (((pooling_block_data[25] & mask) >> i) << 6) |
                                       (((pooling_block_data[26] & mask) >> i) << 5) |
                                       (((pooling_block_data[27] & mask) >> i) << 4) |
                                       (((pooling_block_data[28] & mask) >> i) << 3) |
                                       (((pooling_block_data[29] & mask) >> i) << 2) |
                                       (((pooling_block_data[30] & mask) >> i) << 1) |
                                       (((pooling_block_data[31] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer;
                        cmp_byte_ofs += 4;
                        mask <<= 1;  
                    }
                }
                else
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] / quant_bins[bin_choice];

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer1.x = 0;
                        tmp_buffer1.y = 0;
                        tmp_buffer1.z = 0;
                        tmp_buffer1.w = 0;
                        tmp_buffer2.x = 0;
                        tmp_buffer2.y = 0;
                        tmp_buffer2.z = 0;
                        tmp_buffer2.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer1.x = (((block_data[0] & mask) >> i) << 7) |
                                        (((block_data[1] & mask) >> i) << 6) |
                                        (((block_data[2] & mask) >> i) << 5) |
                                        (((block_data[3] & mask) >> i) << 4) |
                                        (((block_data[4] & mask) >> i) << 3) |
                                        (((block_data[5] & mask) >> i) << 2) |
                                        (((block_data[6] & mask) >> i) << 1) |
                                        (((block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer1.y = (((block_data[8] & mask) >> i) << 7) |
                                        (((block_data[9] & mask) >> i) << 6) |
                                        (((block_data[10] & mask) >> i) << 5) |
                                        (((block_data[11] & mask) >> i) << 4) |
                                        (((block_data[12] & mask) >> i) << 3) |
                                        (((block_data[13] & mask) >> i) << 2) |
                                        (((block_data[14] & mask) >> i) << 1) |
                                        (((block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer1.z = (((block_data[16] & mask) >> i) << 7) |
                                        (((block_data[17] & mask) >> i) << 6) |
                                        (((block_data[18] & mask) >> i) << 5) |
                                        (((block_data[19] & mask) >> i) << 4) |
                                        (((block_data[20] & mask) >> i) << 3) |
                                        (((block_data[21] & mask) >> i) << 2) |
                                        (((block_data[22] & mask) >> i) << 1) |
                                        (((block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer1.w = (((block_data[24] & mask) >> i) << 7) |
                                        (((block_data[25] & mask) >> i) << 6) |
                                        (((block_data[26] & mask) >> i) << 5) |
                                        (((block_data[27] & mask) >> i) << 4) |
                                        (((block_data[28] & mask) >> i) << 3) |
                                        (((block_data[29] & mask) >> i) << 2) |
                                        (((block_data[30] & mask) >> i) << 1) |
                                        (((block_data[31] & mask) >> i) << 0); 
                        
                        // Get i-th bit in 32~39 data.
                        tmp_buffer2.x = (((block_data[32] & mask) >> i) << 7) |
                                        (((block_data[33] & mask) >> i) << 6) |
                                        (((block_data[34] & mask) >> i) << 5) |
                                        (((block_data[35] & mask) >> i) << 4) |
                                        (((block_data[36] & mask) >> i) << 3) |
                                        (((block_data[37] & mask) >> i) << 2) |
                                        (((block_data[38] & mask) >> i) << 1) |
                                        (((block_data[39] & mask) >> i) << 0);
                        
                        // Get i-th bit in 40~47 data.
                        tmp_buffer2.y = (((block_data[40] & mask) >> i) << 7) |
                                        (((block_data[41] & mask) >> i) << 6) |
                                        (((block_data[42] & mask) >> i) << 5) |
                                        (((block_data[43] & mask) >> i) << 4) |
                                        (((block_data[44] & mask) >> i) << 3) |
                                        (((block_data[45] & mask) >> i) << 2) |
                                        (((block_data[46] & mask) >> i) << 1) |
                                        (((block_data[47] & mask) >> i) << 0);

                        // Get i-th bit in 48~55 data.
                        tmp_buffer2.z = (((block_data[48] & mask) >> i) << 7) |
                                        (((block_data[49] & mask) >> i) << 6) |
                                        (((block_data[50] & mask) >> i) << 5) |
                                        (((block_data[51] & mask) >> i) << 4) |
                                        (((block_data[52] & mask) >> i) << 3) |
                                        (((block_data[53] & mask) >> i) << 2) |
                                        (((block_data[54] & mask) >> i) << 1) |
                                        (((block_data[55] & mask) >> i) << 0);

                        // Get i-th bit in 56~63 data.
                        tmp_buffer2.w = (((block_data[56] & mask) >> i) << 7) |
                                        (((block_data[57] & mask) >> i) << 6) |
                                        (((block_data[58] & mask) >> i) << 5) |
                                        (((block_data[59] & mask) >> i) << 4) |
                                        (((block_data[60] & mask) >> i) << 3) |
                                        (((block_data[61] & mask) >> i) << 2) |
                                        (((block_data[62] & mask) >> i) << 1) |
                                        (((block_data[63] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer1;
                        cmp_byte_ofs += 4;
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer2;
                        cmp_byte_ofs += 4;
                        mask <<= 1; 
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}


__global__ void lsCOMP_decompression_kernel_uint16_bsize64(uint16_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ size_t excl_sum;
    __shared__ size_t base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    if(!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    size_t thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.

    // Obtain fixed-rate information for each block.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Obtain block meta data.
            fixed_rate[j] = cmpBytes[block_idx];

            // Check if pooling.
            int pooling_choice = fixed_rate[j] >> 7;
            int temp_rate = fixed_rate[j] & 0x1f;
            if(pooling_choice) thread_ofs += temp_rate * 4;
            else thread_ofs += temp_rate * 8;
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            size_t loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also reading data from global memory.
    size_t base_cmp_byte_ofs = base_idx;
    size_t cmp_byte_ofs;
    size_t tmp_byte_ofs = 0;
    size_t cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;
    
        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Initialization, guiding decoding process.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;

            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            size_t prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                // Buffering decompressed block data.
                uint block_data[64];

                // Read data and shuffle it back from global memory.
                if(pooling_choice)
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) pooling_block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        pooling_block_data[0] |= ((tmp_buffer.x >> 7) & 0x00000001) << i;
                        pooling_block_data[1] |= ((tmp_buffer.x >> 6) & 0x00000001) << i;
                        pooling_block_data[2] |= ((tmp_buffer.x >> 5) & 0x00000001) << i;
                        pooling_block_data[3] |= ((tmp_buffer.x >> 4) & 0x00000001) << i;
                        pooling_block_data[4] |= ((tmp_buffer.x >> 3) & 0x00000001) << i;
                        pooling_block_data[5] |= ((tmp_buffer.x >> 2) & 0x00000001) << i;
                        pooling_block_data[6] |= ((tmp_buffer.x >> 1) & 0x00000001) << i;
                        pooling_block_data[7] |= ((tmp_buffer.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        pooling_block_data[8] |= ((tmp_buffer.y >> 7) & 0x00000001) << i;
                        pooling_block_data[9] |= ((tmp_buffer.y >> 6) & 0x00000001) << i;
                        pooling_block_data[10] |= ((tmp_buffer.y >> 5) & 0x00000001) << i;
                        pooling_block_data[11] |= ((tmp_buffer.y >> 4) & 0x00000001) << i;
                        pooling_block_data[12] |= ((tmp_buffer.y >> 3) & 0x00000001) << i;
                        pooling_block_data[13] |= ((tmp_buffer.y >> 2) & 0x00000001) << i;
                        pooling_block_data[14] |= ((tmp_buffer.y >> 1) & 0x00000001) << i;
                        pooling_block_data[15] |= ((tmp_buffer.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        pooling_block_data[16] |= ((tmp_buffer.z >> 7) & 0x00000001) << i;
                        pooling_block_data[17] |= ((tmp_buffer.z >> 6) & 0x00000001) << i;
                        pooling_block_data[18] |= ((tmp_buffer.z >> 5) & 0x00000001) << i;
                        pooling_block_data[19] |= ((tmp_buffer.z >> 4) & 0x00000001) << i;
                        pooling_block_data[20] |= ((tmp_buffer.z >> 3) & 0x00000001) << i;
                        pooling_block_data[21] |= ((tmp_buffer.z >> 2) & 0x00000001) << i;
                        pooling_block_data[22] |= ((tmp_buffer.z >> 1) & 0x00000001) << i;
                        pooling_block_data[23] |= ((tmp_buffer.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        pooling_block_data[24] |= ((tmp_buffer.w >> 7) & 0x00000001) << i;
                        pooling_block_data[25] |= ((tmp_buffer.w >> 6) & 0x00000001) << i;
                        pooling_block_data[26] |= ((tmp_buffer.w >> 5) & 0x00000001) << i;
                        pooling_block_data[27] |= ((tmp_buffer.w >> 4) & 0x00000001) << i;
                        pooling_block_data[28] |= ((tmp_buffer.w >> 3) & 0x00000001) << i;
                        pooling_block_data[29] |= ((tmp_buffer.w >> 2) & 0x00000001) << i;
                        pooling_block_data[30] |= ((tmp_buffer.w >> 1) & 0x00000001) << i;
                        pooling_block_data[31] |= ((tmp_buffer.w >> 0) & 0x00000001) << i;
                    }

                    // Assign data back to block data.
                    for(int i=0; i<32; i++)
                    {
                        block_data[i*2] = pooling_block_data[i] * quant_bins[bin_choice];
                        block_data[i*2+1] = block_data[i*2];
                    }
                }
                else
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer1 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;
                        tmp_buffer2 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        block_data[0] |= ((tmp_buffer1.x >> 7) & 0x00000001) << i;
                        block_data[1] |= ((tmp_buffer1.x >> 6) & 0x00000001) << i;
                        block_data[2] |= ((tmp_buffer1.x >> 5) & 0x00000001) << i;
                        block_data[3] |= ((tmp_buffer1.x >> 4) & 0x00000001) << i;
                        block_data[4] |= ((tmp_buffer1.x >> 3) & 0x00000001) << i;
                        block_data[5] |= ((tmp_buffer1.x >> 2) & 0x00000001) << i;
                        block_data[6] |= ((tmp_buffer1.x >> 1) & 0x00000001) << i;
                        block_data[7] |= ((tmp_buffer1.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        block_data[8] |= ((tmp_buffer1.y >> 7) & 0x00000001) << i;
                        block_data[9] |= ((tmp_buffer1.y >> 6) & 0x00000001) << i;
                        block_data[10] |= ((tmp_buffer1.y >> 5) & 0x00000001) << i;
                        block_data[11] |= ((tmp_buffer1.y >> 4) & 0x00000001) << i;
                        block_data[12] |= ((tmp_buffer1.y >> 3) & 0x00000001) << i;
                        block_data[13] |= ((tmp_buffer1.y >> 2) & 0x00000001) << i;
                        block_data[14] |= ((tmp_buffer1.y >> 1) & 0x00000001) << i;
                        block_data[15] |= ((tmp_buffer1.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        block_data[16] |= ((tmp_buffer1.z >> 7) & 0x00000001) << i;
                        block_data[17] |= ((tmp_buffer1.z >> 6) & 0x00000001) << i;
                        block_data[18] |= ((tmp_buffer1.z >> 5) & 0x00000001) << i;
                        block_data[19] |= ((tmp_buffer1.z >> 4) & 0x00000001) << i;
                        block_data[20] |= ((tmp_buffer1.z >> 3) & 0x00000001) << i;
                        block_data[21] |= ((tmp_buffer1.z >> 2) & 0x00000001) << i;
                        block_data[22] |= ((tmp_buffer1.z >> 1) & 0x00000001) << i;
                        block_data[23] |= ((tmp_buffer1.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        block_data[24] |= ((tmp_buffer1.w >> 7) & 0x00000001) << i;
                        block_data[25] |= ((tmp_buffer1.w >> 6) & 0x00000001) << i;
                        block_data[26] |= ((tmp_buffer1.w >> 5) & 0x00000001) << i;
                        block_data[27] |= ((tmp_buffer1.w >> 4) & 0x00000001) << i;
                        block_data[28] |= ((tmp_buffer1.w >> 3) & 0x00000001) << i;
                        block_data[29] |= ((tmp_buffer1.w >> 2) & 0x00000001) << i;
                        block_data[30] |= ((tmp_buffer1.w >> 1) & 0x00000001) << i;
                        block_data[31] |= ((tmp_buffer1.w >> 0) & 0x00000001) << i;

                        // Get ith bit in 32~39 abs quant from global memory.
                        block_data[32] |= ((tmp_buffer2.x >> 7) & 0x00000001) << i;
                        block_data[33] |= ((tmp_buffer2.x >> 6) & 0x00000001) << i;
                        block_data[34] |= ((tmp_buffer2.x >> 5) & 0x00000001) << i;
                        block_data[35] |= ((tmp_buffer2.x >> 4) & 0x00000001) << i;
                        block_data[36] |= ((tmp_buffer2.x >> 3) & 0x00000001) << i;
                        block_data[37] |= ((tmp_buffer2.x >> 2) & 0x00000001) << i;
                        block_data[38] |= ((tmp_buffer2.x >> 1) & 0x00000001) << i;
                        block_data[39] |= ((tmp_buffer2.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 40~47 abs quant from global memory.
                        block_data[40] |= ((tmp_buffer2.y >> 7) & 0x00000001) << i;
                        block_data[41] |= ((tmp_buffer2.y >> 6) & 0x00000001) << i;
                        block_data[42] |= ((tmp_buffer2.y >> 5) & 0x00000001) << i;
                        block_data[43] |= ((tmp_buffer2.y >> 4) & 0x00000001) << i;
                        block_data[44] |= ((tmp_buffer2.y >> 3) & 0x00000001) << i;
                        block_data[45] |= ((tmp_buffer2.y >> 2) & 0x00000001) << i;
                        block_data[46] |= ((tmp_buffer2.y >> 1) & 0x00000001) << i;
                        block_data[47] |= ((tmp_buffer2.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 48-55 abs quant from global memory.
                        block_data[48] |= ((tmp_buffer2.z >> 7) & 0x00000001) << i;
                        block_data[49] |= ((tmp_buffer2.z >> 6) & 0x00000001) << i;
                        block_data[50] |= ((tmp_buffer2.z >> 5) & 0x00000001) << i;
                        block_data[51] |= ((tmp_buffer2.z >> 4) & 0x00000001) << i;
                        block_data[52] |= ((tmp_buffer2.z >> 3) & 0x00000001) << i;
                        block_data[53] |= ((tmp_buffer2.z >> 2) & 0x00000001) << i;
                        block_data[54] |= ((tmp_buffer2.z >> 1) & 0x00000001) << i;
                        block_data[55] |= ((tmp_buffer2.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 56-63 abs quant from global memory.
                        block_data[56] |= ((tmp_buffer2.w >> 7) & 0x00000001) << i;
                        block_data[57] |= ((tmp_buffer2.w >> 6) & 0x00000001) << i;
                        block_data[58] |= ((tmp_buffer2.w >> 5) & 0x00000001) << i;
                        block_data[59] |= ((tmp_buffer2.w >> 4) & 0x00000001) << i;
                        block_data[60] |= ((tmp_buffer2.w >> 3) & 0x00000001) << i;
                        block_data[61] |= ((tmp_buffer2.w >> 2) & 0x00000001) << i;
                        block_data[62] |= ((tmp_buffer2.w >> 1) & 0x00000001) << i;
                        block_data[63] |= ((tmp_buffer2.w >> 0) & 0x00000001) << i;
                    }

                    // Restore quantized data.
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] * quant_bins[bin_choice];
                }

                // Write data back to global memory.
                data_idx_x = block_idx_x;
                for(uint i=0; i<8; i++)
                {
                    data_idx_y = block_idx_y * 8 + i;
                    for(uint k=0; k<8; k++)
                    {
                        data_idx_z = block_idx_z * 8 + k;
                        data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                        if(data_idx_y < dims.y && data_idx_z < dims.z) decData[data_idx] = block_data[i*8+k];
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}