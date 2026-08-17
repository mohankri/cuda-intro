#include <iostream>
#include "cuda_runtime.h"
#include "nccl.h"

using namespace std;

static int attr(int dev, cudaDeviceAttr a) {
    int v = 0; cudaDeviceGetAttribute(&v, a, dev); return v;
}

int
main(int argc, char **argv) {
    int nDev = 0;
    cudaError_t err = cudaGetDeviceCount(&nDev);
    cout << "Number of Device " << nDev << endl;

    for  (int d = 0; d < nDev; d++) {
        cout << " SMP Count " << attr(d, cudaDevAttrMultiProcessorCount) << endl;
        cout << " Number of Thread per SMP " << attr(d, cudaDevAttrMaxThreadsPerMultiProcessor) << endl;
        cout << " Max Block per SMP " << attr(d, cudaDevAttrMaxBlocksPerMultiprocessor) << endl;
        cout << " Register per SM " << attr(d, cudaDevAttrMaxRegistersPerMultiprocessor) << endl;
        cout << " Shared Memory per SM " << attr(d, cudaDevAttrMaxSharedMemoryPerMultiprocessor) << endl;
        cout << " Shared Memory per Block " << attr(d, cudaDevAttrMaxSharedMemoryPerBlockOptin) << endl;
        cout << " Warp size " << attr(d, cudaDevAttrWarpSize) << endl;
        cout << " Clock Rate " << attr(d, cudaDevAttrClockRate) / 1e6 << endl;

        int warpCap  = attr(d, cudaDevAttrMaxThreadsPerMultiProcessor) / attr(d, cudaDevAttrWarpSize);
        cout << " Warp Capacity " << warpCap << endl;
    }

    return 0;
}