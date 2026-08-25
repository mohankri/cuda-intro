#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

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

int main(int argc, char** argv) {
    const size_t elementCount = argc > 1
        ? std::strtoull(argv[1], nullptr, 10)
        : 1 << 22;  // 16M bytes per GPU by default.
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
    if (elementCount == 0 || iterations <= 0) {
        std::fprintf(stderr, "Usage: %s [element_count] [iterations]\n", argv[0]);
        return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    std::vector<int> devices(deviceCount);
    std::vector<ncclComm_t> communicators(deviceCount);
    std::vector<cudaStream_t> streams(deviceCount);
    std::vector<cudaEvent_t> startEvents(deviceCount);
    std::vector<cudaEvent_t> stopEvents(deviceCount);
    std::vector<float*> deviceInputs(deviceCount);
    std::vector<float*> deviceOutputs(deviceCount);

    for (int rank = 0; rank < deviceCount; ++rank) {
        devices[rank] = rank;
    }

    // Initialize the communicator once; its setup cost is excluded from timing.
    NCCL_CHECK(ncclCommInitAll(communicators.data(), deviceCount, devices.data()));

    std::vector<float> hostValues(elementCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaStreamCreate(&streams[rank]));
        CUDA_CHECK(cudaEventCreate(&startEvents[rank]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[rank]));
        CUDA_CHECK(cudaMalloc(&deviceInputs[rank], elementCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&deviceOutputs[rank], elementCount * sizeof(float)));
        std::fill(hostValues.begin(), hostValues.end(), static_cast<float>(rank + 1));
        CUDA_CHECK(cudaMemcpyAsync(deviceInputs[rank], hostValues.data(),
                                   elementCount * sizeof(float), cudaMemcpyHostToDevice,
                                   streams[rank]));
    }

    // Warm up NCCL so one-time algorithm and transport setup is not measured.
    ncclGroupStart();
    for (int rank = 0; rank < deviceCount; ++rank) {
        NCCL_CHECK(ncclAllReduce(deviceInputs[rank], deviceOutputs[rank], elementCount,
                                 ncclFloat, ncclSum, communicators[rank], streams[rank]));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }

    // Group all ranks for every iteration so NCCL can schedule the operation together.
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaEventRecord(startEvents[rank], streams[rank]));
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        ncclGroupStart();
        for (int rank = 0; rank < deviceCount; ++rank) {
            NCCL_CHECK(ncclAllReduce(deviceInputs[rank], deviceOutputs[rank], elementCount,
                                     ncclFloat, ncclSum, communicators[rank],
                                     streams[rank]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaEventRecord(stopEvents[rank], streams[rank]));
    }
    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaEventSynchronize(stopEvents[rank]));
    }

    float elapsedMilliseconds = 0.0f;
    for (int rank = 0; rank < deviceCount; ++rank) {
        float rankMilliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&rankMilliseconds, startEvents[rank],
                                        stopEvents[rank]));
        if (rankMilliseconds > elapsedMilliseconds) {
            elapsedMilliseconds = rankMilliseconds;
        }
    }

    const float expected = deviceCount * (deviceCount + 1) / 2.0f;
    float firstValue = 0.0f;
    CUDA_CHECK(cudaSetDevice(devices[0]));
    CUDA_CHECK(cudaMemcpy(&firstValue, deviceOutputs[0], sizeof(float),
                          cudaMemcpyDeviceToHost));

    const double bytesPerIteration = static_cast<double>(elementCount) * sizeof(float) *
                                     2.0 * (deviceCount - 1) / deviceCount;
    const double bandwidthGBs = bytesPerIteration * iterations /
                                (elapsedMilliseconds * 1.0e6);
    std::printf("NCCL AllReduce: %d GPUs, %zu floats, %d iterations\n",
                deviceCount, elementCount, iterations);
    std::printf("First result: %.1f (expected %.1f)\n", firstValue, expected);
    std::printf("Average time: %.3f ms, algorithm bandwidth: %.2f GB/s\n",
                elapsedMilliseconds / iterations, bandwidthGBs);

    for (int rank = 0; rank < deviceCount; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        CUDA_CHECK(cudaFree(deviceInputs[rank]));
        CUDA_CHECK(cudaFree(deviceOutputs[rank]));
        CUDA_CHECK(cudaEventDestroy(startEvents[rank]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[rank]));
        CUDA_CHECK(cudaStreamDestroy(streams[rank]));
        NCCL_CHECK(ncclCommDestroy(communicators[rank]));
    }
    return firstValue == expected ? 0 : 1;
}
