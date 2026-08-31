#include "cuda_runtime.h"
#include "nccl.h"
#include <iostream>

using namespace std;

__global__ void add_10_to_each(int *arr, int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = arr[idx] + 10;
}

int
main(int argc, char **argv) {
    int nDev = 0;
    cudaError_t err = cudaGetDeviceCount(&nDev);
    std::cout << "Number of Device " << nDev << std::endl;
    int *d_a = nullptr;
    void *d_out = nullptr;

    cudaMalloc((void **)&d_a, 10 * sizeof(int));
    cudaMalloc((void **)&d_out, 10 * sizeof(int));

    add_10_to_each<<<1, 10>>>(d_a, d_out);
    int h_out[10];
    cudaMemcpy(h_out, d_out, 10 * sizoeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
