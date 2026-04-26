
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstdint>

__global__ void fused_aq_packed_kernel(
    const uint8_t* __restrict__ packed_indices,
    const half* __restrict__ codebooks,
    const half* __restrict__ inputs,
    half* __restrict__ outputs,
    int num_codebooks, 
    int dict_size,     
    int in_features,   
    int out_features,
    int stride, 
    int total_codebook_elements,
    int num_tokens
) {
    extern __shared__ half shared_data[];
    half* shared_codebooks = shared_data;
    // Offset by codebook size to create the LUT
    int8_t* decode_lut = (int8_t*)&shared_data[total_codebook_elements];

    int tid = threadIdx.x;

    // 1. Load Codebooks cooperatively
    for (int i = tid; i < total_codebook_elements; i += blockDim.x) {
        shared_codebooks[i] = codebooks[i];
    }
    
    // 🚨 FIX 1: Safe LUT Init (Works even if threads < 243)
    for (int i = tid; i < 243; i += blockDim.x) {
        uint8_t val = i;
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            decode_lut[i * 5 + j] = (int8_t)(val % 3) - 1;
            val /= 3;
        }
    }
    __syncthreads();

    // 2. Token & Row Mapping
    int token_idx = blockIdx.x; 
    int row = blockIdx.y * blockDim.x + threadIdx.x;

    if (row < out_features && token_idx < num_tokens) {
        float total_sum = 0.0f;
        
        uint64_t row_offset = (uint64_t)row * stride * num_codebooks;
        const half* my_input = inputs + ((uint64_t)token_idx * in_features);

        for (int col_block = 0; col_block < stride; ++col_block) {
            float weights5[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            for (int m = 0; m < num_codebooks; ++m) {
                uint8_t packed_byte = packed_indices[row_offset + (col_block * num_codebooks) + m];

                 if (packed_byte < 243) {
                    #pragma unroll
                    for (int i = 0; i < 5; i++) {
                        int8_t ternary_val = decode_lut[packed_byte * 5 + i];
                        
                        if (ternary_val != 0) {
                            weights5[i] += __half2float(shared_codebooks[m * dict_size + ((ternary_val == -1) ? 0 : 1)]);
                        }
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < 5; i++) {
                int actual_col = col_block * 5 + i;
                if (actual_col < in_features) {
                    total_sum += weights5[i] * __half2float(my_input[actual_col]);
                }
            }
        }
        outputs[(uint64_t)token_idx * out_features + row] = __float2half(total_sum);
    }
}

void launch_aq_packed_kernel(
    const uint8_t* indices, const at::Half* codebooks, const at::Half* inputs, at::Half* outputs,
    int blocks_y,
    int threads, int shared_mem_size,
    int num_codebooks, int dict_size, int in_features, int out_features, int stride, 
    int total_codebook_elements, int num_tokens
) {
    dim3 threads_per_block(threads);
    dim3 grid(num_tokens, blocks_y); 

    fused_aq_packed_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        (const uint8_t*)indices, (const half*)codebooks, (const half*)inputs, (half*)outputs, 
        num_codebooks, dict_size, in_features, out_features, stride, 
        total_codebook_elements, num_tokens
    );
}
