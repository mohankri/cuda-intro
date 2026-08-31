#include "cuda_runtime.h"
#include "nccl.h"
#include <iostream>

using namespace std;

#define CUDA_CHECK(call) \
    do {   \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(error)); \
            return 1; \
        } \
    } while (0)

__global__ void launch_kernel() {
    // This kernel does nothing, it's just for testing launch parameters
}

int
main(int argc, char **argv) {
    int nDev = 0;

    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    std::cout << "Number of GPU " << nDev << std::endl;
    int val = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount , 0));
    std::fprintf(stderr, "cudaDevAttrMultiProcessorCount: %d\n", val);

    int val1 = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&val1, cudaDevAttrMaxThreadsPerMultiProcessor , 0));
    std::fprintf(stderr, "cudaDevAttrMaxThreadsPerMultiProcessor" ": %d\n", val);

    std::fprintf(stderr, "Max Threads for whole GPU = %d\n", val * val1);

    val = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlocksPerMultiprocessor , 0));
    std::fprintf(stderr, "cudaDevAttrMaxBlocksPerMultiprocessor = %d\n", val);

    val = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerBlock , 0));
    std::fprintf(stderr, "cudaDevAttrMaxThreadsPerBlock = %d\n", val);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    launch_kernel<<<1, 2048>>>();
    cudaError_t lastErr = cudaGetLastError();
    std::cerr << "Launch kernel error: " << cudaGetErrorString(lastErr) << std::endl;

    launch_kernel<<<1, 1024>>>();
    lastErr = cudaGetLastError();
    std::cerr << "Launch kernel error: " << cudaGetErrorString(lastErr) << std::endl;

    launch_kernel<<<1, 512>>>();
    lastErr = cudaGetLastError();
    std::cerr << "Launch kernel error: " << cudaGetErrorString(lastErr) << std::endl;

    return 0;
}