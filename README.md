```
SW/HW mapping
```
<img width="1472" height="800" alt="image" src="https://github.com/user-attachments/assets/8bb95267-c7ac-4503-bc08-7b6d203b2b7a" />

```
$ nvcc hello.cu -o hello -lcudart -lnccl

$ ~/cuda-intro$ ./hello

Number of Device 1
 SMP Count 72
 Number of Thread per SMP 1536
 Max Block per SMP 16
 Register per SM 65536
 Shared Memory per SM 102400
 Shared Memory per Block 101376
 Warp size (Number of threads per Warp) 32
 Clock Rate 1.695
 Warp Capacity 48
```
```
SMP Count on 1 GPU (SMP Count 72)
```
<img width="1190" height="630" alt="image" src="https://github.com/user-attachments/assets/7af24622-5355-4168-807c-eaee6100d5af" />

```
Number of Block per SMP (Max Block per SMP 16)
```
<img width="1190" height="596" alt="image" src="https://github.com/user-attachments/assets/ae203918-7ab2-4add-9363-dbdceda718de" />


```
Number of warps per SM (warp capacity 48)
Most GPUs have four warp schedulers per SM.
```
<img width="1190" height="666" alt="image" src="https://github.com/user-attachments/assets/766df967-378e-44a4-b635-1ca7db373348" />


```
1 Warp Slot (number of threads per warp: 32)
```
<img width="1190" height="578" alt="image" src="https://github.com/user-attachments/assets/beeba8db-e655-433c-87a1-39e7d56b132d" />

```
Kernel Launch <<<1, 1024>>>
```
<img width="1190" height="980" alt="image" src="https://github.com/user-attachments/assets/2e78ff0f-4271-4b7a-a1f6-e1d6dd61b885" />

```
Kernel Launch <<< 72, 1024>>>
```
<img width="1190" height="980" alt="image" src="https://github.com/user-attachments/assets/8b1ff4c7-221f-4662-a1e8-b85bb916c4d2" />

```
Kernel Launch <<< 1, 64>>>
will use 1 SM and 2 warps (32 threads each)
```
<img width="1190" height="946" alt="image" src="https://github.com/user-attachments/assets/f49154c2-41ed-46f9-932c-a0485fbe6eca" />

```
Kernel Launch <<<256, 512>>>

## Occupancy accounting: `<<<256, 512>>>` on NVIDIA A10 (sm_86)

| Level               | Count   | Derivation                                    |
|---------------------|---------|-----------------------------------------------|
| Total threads       | 131,072 | 256 blocks × 512 threads                      |
| Total warps         | 4,096   | 131,072 / 32                                  |
| Warps per block     | 16      | 512 / 32                                      |
| Blocks per SM       | 3       | ⌊1536 / 512⌋ — thread limit binds             |
| Warps per SM        | 48      | 3 × 16 = SM max → 100% theoretical occupancy  |
| Warps per scheduler | 12      | 48 / 4 sub-partitions                         |
| Waves               | ~1.19   | 256 / (72 SMs × 3 blocks)                     |

### Resource thresholds to sustain 3 blocks/SM

| Resource       | SM pool  | Threshold per block      | If exceeded          |
|----------------|----------|--------------------------|----------------------|
| Registers      | 65,536   | ≤ 42 regs/thread         | 2 blocks → 66% occ.  |
| Shared memory  | 102,400 B| ≤ ~33 KB                 | 2 blocks → 66% occ.  |
| Block slots    | 16       | n/a (3 ≪ 16)             | not binding          |

```

<img width="1472" height="960" alt="image" src="https://github.com/user-attachments/assets/9ce54bed-1599-4cb2-884b-d6aadc095889" />

```
Basic understanding of
blockDim.x, blockId.x and threadIdx.x

```
<img width="1220" height="394" alt="image" src="https://github.com/user-attachments/assets/485a5afc-0074-4736-9376-d81439b36739" />

```
Basic understanding of
blockDim.y, blockId.y and threadIdx.y
```
<img width="1472" height="1400" alt="image" src="https://github.com/user-attachments/assets/87394913-5e36-4ac5-ae02-f6b493469f0d" />

## NVIDIA GPU topology

Captured with `nvidia-smi topo -m` on four NVIDIA GB300 GPUs. Every GPU pair is connected by `NV18`.

```mermaid
graph TD
	CPU0[CPU NUMA 0<br/>CPU affinity 0-71]
	CPU1[CPU NUMA 1<br/>CPU affinity 72-143]

	subgraph N0[GPU NUMA locality]
		GPU0[GPU0<br/>GPU NUMA 2]
		GPU1[GPU1<br/>GPU NUMA 10]
	end

	subgraph N1[GPU NUMA locality]
		GPU2[GPU2<br/>GPU NUMA 18]
		GPU3[GPU3<br/>GPU NUMA 26]
	end

	CPU0 -. CPU affinity .- GPU0
	CPU0 -. CPU affinity .- GPU1
	CPU1 -. CPU affinity .- GPU2
	CPU1 -. CPU affinity .- GPU3

	GPU0 ---|NV18| GPU1
	GPU0 ---|NV18| GPU2
	GPU0 ---|NV18| GPU3
	GPU1 ---|NV18| GPU2
	GPU1 ---|NV18| GPU3
	GPU2 ---|NV18| GPU3
```

`NV18` means a bonded set of 18 NVLinks. `X` on the `nvidia-smi topo -m` diagonal is the GPU's self-connection.
