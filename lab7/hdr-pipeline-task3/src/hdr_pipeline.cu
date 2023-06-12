#include <cstdint>
#include "utils/cuda/error.h"

#include "bloom_kernel.cuh"

unsigned int divup(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


__device__ float tonemap(float value, float exposure) {
    float v = value * exposure;
    return (v * (0.9036f * v + 0.018f)) / (v * (0.8748f * v + 0.354f) + 0.14f);
}

__device__ float3 tonemap(float3 value, float exposure) {
    return { tonemap(value.x, exposure), tonemap(value.y, exposure), tonemap(value.z, exposure) };
}

__device__ float sRGB8(float color) {
    return color <= 0.0031308f ? color * 12.92f : 1.055f * pow(color, 1.0f / 2.4f) - 0.055f;
}

__device__ uint32_t norm(float color) {
    return static_cast<uint32_t>(min(max(color, 0.0f), 1.0f) * 255.0f);
}

__device__ uint32_t sRGB8(float3 color) {
    uint32_t r = norm(sRGB8(color.x));
    uint32_t g = norm(sRGB8(color.y));
    uint32_t b = norm(sRGB8(color.z));
    uint32_t a = 255;
    return (a << 24) | (b << 16) | (g << 8) | r;
}

__global__ void tonemap_kernel(uint32_t* out, const float* in, int width, int height, float exposure) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height) {
        float3 color = { in[4*(y*width + x) + 0], in[4*(y*width + x) + 1], in[4*(y*width + x) + 2] };

        uint32_t srgb = sRGB8(tonemap(color, exposure));

        out[y*width + x] = srgb; 
    }
}

__device__ float log_avg_lum;

__device__ float luminance(float3 color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

__global__ void log_avg_lum_kernel(const float* in, int width, int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float3 color = { in[4 * (y * width + x) + 0], in[4 * (y * width + x) + 1], in[4 * (y * width + x) + 2] };

        float lum = luminance(color);

        atomicAdd(&log_avg_lum, log(lum));
    }
}

__device__ float calc_fade_contribution(float3 v, float exposure, float threshold){
    
    

    float res = powf(saturate((tonemap(v, exposure) - 0.8f * threshold) / 0.2f * threshold), 2);
    return res;
}

float saturate(float x){
    if(x < 0.0f){
        return 0.0f;
    } 
    if (x > 1.0f){
        return 1.0f;
    }
    else{
        return x;
    }
}


void bright_pass_x_y(float exposure, const float* in){
    const float threshold = 1.0f;
    // todo: cuda e
    calc_fade_contribution(in, exposure, threshold) * in;
}


void blurr(){

}

void tonemap(uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold) {
    int block_size_x = 32;
    int block_size_y = 8;

    dim3 block(block_size_x, block_size_y, 1);
    dim3 grid(divup(width, block.x), divup(height, block.y), 1);
    tonemap_kernel<<<grid, block>>>(out, in, width, height, exposure);
    throw_error(cudaDeviceSynchronize());
    throw_error(cudaPeekAtLastError());
}

float compute_avg_luminance(const float* in, int width, int height) {
    constexpr int block_size_x = 128;
    constexpr int block_size_y = 8;

    float luminance = 0;

    const dim3 block_size = { block_size_x , block_size_y };
    const dim3 num_blocks = { divup(width, block_size_x), divup(height, block_size_y) };

    cudaMemcpyToSymbol(log_avg_lum, &luminance, sizeof(float));

    log_avg_lum_kernel<<<num_blocks, block_size>>>(in, width, height);

    cudaMemcpyFromSymbol(&luminance, log_avg_lum, sizeof(float));

    return std::exp(luminance / (width * height));
}
