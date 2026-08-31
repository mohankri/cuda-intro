#include <iostream>
#include "cuda_runtime.h"

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


__global__ void saxpy() {

}

int
main(int argc, char **argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 10;
    const float a = 2.0f;
    const size_t bytes = n * sizeof(float);
    float *h_x = nullptr;
    float *h_y = nullptr;

    /* host buffer, pinned fast DMA */
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));

    for (int i = 0; i < bytes; i++) {
        h_x[i] = 1.0f;
        h_y[i] = (float)(i % 7);
    }

    float *d_x, *d_y;
    /* device buffer allocated on the GPU */
    CUDA_CHECK(cudaMalloc((void **)&d_x, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y, bytes));

    /* Copy Data to Device Buffer */
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    /* Calculate resource on GPU to perform SAXPY */
    int block = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&block, cudaDevAttrMaxBlocksPerMultiprocessor , 0));
    std::fprintf(stderr, "Max Block for SMP %d\n", block);

    int warpSize = 0; // number of threads in a warp
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0));
    std::fprintf(stderr, "Warp size %d\n", warpSize);

}