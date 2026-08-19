#include <stdio.h>
#include "cuda_runtime.h"

__global__ void matrix_multiply(int *A, int *B, int *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> column (fast dim)
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y -> row

    if (row < n && col < n) {
        int acc = 0;
        for (int k = 0; k < n; k++)
            acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

int main(int argc, char **argv) {
    
    const int MATRIX_SIZE = 4;
    const int MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
    
    // Initialize 4x4 matrices A and B
    const int A[MATRIX_ELEMENTS] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    const int B[MATRIX_ELEMENTS] = {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4
    };
    
    int C[MATRIX_ELEMENTS] = {0};

    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;

    cudaMalloc((void**)&dev_a, MATRIX_ELEMENTS * sizeof(int));
    cudaMalloc((void**)&dev_b, MATRIX_ELEMENTS * sizeof(int));
    cudaMalloc((void**)&dev_c, MATRIX_ELEMENTS * sizeof(int));

    cudaMemcpy(dev_a, A, MATRIX_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B, MATRIX_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Launch: the whole 4x4 fits in one 4x4 block = 16 threads
    dim3 block(MATRIX_SIZE, MATRIX_SIZE);
    dim3 grid(1, 1);

    matrix_multiply<<<grid, block>>>(dev_a, dev_b, dev_c, MATRIX_SIZE);

    cudaMemcpy(C, dev_c, MATRIX_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result Matrix C (4x4):\n");
    for (int i = 0; i < MATRIX_ELEMENTS; i++) {
        printf("%d ", C[i]);
        if ((i + 1) % MATRIX_SIZE == 0) printf("\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
