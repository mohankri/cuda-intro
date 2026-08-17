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
 Warp size 32
 Clock Rate 1.695
 Warp Capacity 48
```
```
SMP Count on 1 GPU
```
<img width="844" height="472" alt="image" src="https://github.com/user-attachments/assets/f84e1ccb-5a0d-4610-b561-9e946193b7c1" />

