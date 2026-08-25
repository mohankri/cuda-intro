#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;
/*
 * Simple NCCL multi-GPU example.
 * Each GPU contributes its rank plus one. ncclAllReduce performs a sum
 * across all GPUs, so every GPU receives the same total value. The program
 * creates one NCCL communicator and CUDA stream per GPU, launches the
 * collective, then copies and prints the result from each device.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(error)); \
            return 1; \
        } \
    } while (0)

#define NCCL_CHECK(call) \
    do { \
        ncclResult_t result = (call); \
        if (result != ncclSuccess) { \
            std::fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, \
                         ncclGetErrorString(result)); \
            return 1; \
        } \
    } while (0)

int main() {
    // Discover the GPUs available to participate in the collective operation.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }
    std::cout << "NCCL AllReduce across " << deviceCount << " GPUs:\n";

    std::vector<int> devices(deviceCount);
    std::vector<ncclComm_t> communicators(deviceCount);
    std::vector<cudaStream_t> streams(deviceCount);
    std::vector<float*> deviceValues(deviceCount);
    std::vector<float> results(deviceCount);

    // Map each NCCL rank to one GPU device.
    for (int rank = 0; rank < deviceCount; ++rank) {
        devices[rank] = rank;
    }

    // Create one NCCL communicator for the group of participating GPUs.
    NCCL_CHECK(ncclCommInitAll(communicators.data(), deviceCount, devices.data()));

    // Create a stream and device buffer for each GPU, then initialize its
    // contribution with the value rank + 1.
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaStreamCreate(&streams[rank]));
        CUDA_CHECK(cudaMalloc(&deviceValues[rank], sizeof(float)));

        float value = static_cast<float>(rank + 1);
        CUDA_CHECK(cudaMemcpy(deviceValues[rank], &value, sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    /*
     * Perform an in-place AllReduce on every GPU. Each rank contributes its
     * local value (rank + 1); NCCL sums those values across the communicator
     * and writes the same total back to every device buffer. The collective
     * is enqueued asynchronously on the rank's CUDA stream and completes
     * before the synchronization and host copy below.
     */
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));

        NCCL_CHECK(ncclAllReduce(deviceValues[rank], deviceValues[rank], 1,
                                ncclFloat, ncclSum, communicators[rank],
                                streams[rank]));
    }

    // Wait for each collective to finish before copying its result to the host.
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
        CUDA_CHECK(cudaMemcpy(&results[rank], deviceValues[rank], sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Confirm that every GPU received the expected sum of all contributions.
    float expected = deviceCount * (deviceCount + 1) / 2.0f;
    std::printf("NCCL AllReduce across %d GPUs:\n", deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        std::printf("  GPU %d: %.1f (expected %.1f)\n", rank, results[rank], expected);
    }

    // Release device memory, CUDA streams, and NCCL communicators.
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaFree(deviceValues[rank]));
        CUDA_CHECK(cudaStreamDestroy(streams[rank]));
        NCCL_CHECK(ncclCommDestroy(communicators[rank]));
    }

    std::cout << "NCCL AllReduce completed successfully.\n";
    return 0;
}
