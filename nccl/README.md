# NCCL AllReduce

This folder contains multi-GPU NCCL examples. The main grouped example is
[`nccl_performant_allreduce.cu`](nccl_performant_allreduce.cu).

## Run the grouped example

```bash
nvcc nccl_performant_allreduce.cu -o nccl_performant_allreduce -lcudart -lnccl
./nccl_performant_allreduce
```

Each GPU contributes one value, `rank + 1`. With four GPUs, the AllReduce
sum is `1 + 2 + 3 + 4 = 10`, and every GPU receives `10` in its local buffer.

## NCCL internals during execution

The program creates one communicator and one CUDA stream per GPU. Because one
host thread launches collectives for multiple communicators, the calls are
wrapped in `ncclGroupStart()` and `ncclGroupEnd()`. This lets NCCL coordinate
the launches as a single grouped operation.

```mermaid
flowchart TD
    H[Host thread]
    D[Discover GPU count]
    C[ncclCommInitAll<br/>Create communicator per GPU]
    B[Allocate one device buffer<br/>and CUDA stream per GPU]
    I[Initialize local values<br/>GPU 0 = 1, GPU 1 = 2, ...]
    GS[ncclGroupStart]
    A[ncclAllReduce on every rank<br/>in-place deviceValues[rank]]
    GE[ncclGroupEnd]
    T[NCCL transport selection<br/>topology-aware rings or trees]
    RS[Logical reduction phase<br/>combine values across GPUs]
    AG[Logical distribution phase<br/>make the final sum available on every GPU]
    S[cudaStreamSynchronize<br/>wait for NCCL work]
    Q[Copy one result per GPU<br/>device to host]
    V[Verify expected sum<br/>and release resources]

    H --> D --> C --> B --> I --> GS --> A --> GE
    GE --> T --> RS --> AG --> S --> Q --> V
    A -. launches work on .-> S0[GPU 0 CUDA stream]
    A -. launches work on .-> S1[GPU 1 CUDA stream]
    A -. launches work on .-> SN[GPU N CUDA stream]
    S0 --> T
    S1 --> T
    SN --> T
```

The reduction and distribution phases are a logical view of AllReduce. NCCL
selects the concrete communication algorithm and path based on the hardware
topology, message size, and available transport links. The operation is
asynchronous when queued on each CUDA stream; the later stream synchronization
makes the device results safe to copy back to the host.

## Source files

- `nccl_allreduce.cu`: basic one-value AllReduce example.
- `nccl_performant_allreduce.cu`: grouped AllReduce launch using
  `ncclGroupStart()` and `ncclGroupEnd()`.
- `nccl_reduce.cu`: larger repeated AllReduce benchmark with CUDA event timing.
