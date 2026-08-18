#include <iostream>
#include <stdio.h>

#include "cuda_runtime.h"
#include "nccl.h"

using namespace std;

static int attr(int dev, cudaDeviceAttr a) {
    int v = 0; cudaDeviceGetAttribute(&v, a, dev); return v;
}

__global__ void probe(unsigned* blockSm) {

}

static void tryLaunch(int grid, int block) {
    printf("<<<%d, %d>>>  ", grid, block);
    unsigned * dSm = nullptr;

    probe<<<grid, block>>>(dSm);
    cudaError_t launchErr = cudaGetLastError();      // launch-time legality
    if (launchErr != cudaSuccess) {
        cout << "Launch failed " << block << " " << endl;
    } else {
        cout << "Launch Sucess " << block << " " << endl;
    }
    return;
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
        cout << " Warp size (Number of threads per Warp) " << attr(d, cudaDevAttrWarpSize) << endl;
        cout << " Max Thread per Block " << attr(d, cudaDevAttrMaxThreadsPerBlock) << endl;
        cout << " Clock Rate " << attr(d, cudaDevAttrClockRate) / 1e6 << endl;

        int warpCap  = attr(d, cudaDevAttrMaxThreadsPerMultiProcessor) / attr(d, cudaDevAttrWarpSize);
        cout << " Warp Capacity " << warpCap << endl;
    }

    cout << " Kernel Launch " << endl;

    tryLaunch(1,   1024);   // legal, one SM, 67% there
    tryLaunch(1,   1025);   // fails: over block cap
    tryLaunch(1,   1536);   // fails: SM cap doesn't help
    tryLaunch(2,   1024);   // two blocks -> two SMs
    tryLaunch(72, 1024);   // one per SM, 67% everywhere
    tryLaunch(4 * 72, 384);// 4 per SM, 48/48 = 100%


    return 0;
}