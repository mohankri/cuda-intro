/*
 * Vector Addition using CUDA
 * 
 * This program demonstrates basic CUDA programming by performing element-wise
 * addition of two arrays on the GPU. It allocates memory on the GPU, copies
 * input arrays from host to device, launches a parallel kernel to compute
 * the sum, and copies the result back to the host for display.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int
main(int argc, char **argv) {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;

    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, arraySize>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: ");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}