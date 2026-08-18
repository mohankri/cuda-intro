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
Number of Warp per SMP (Warp size 48)
Most of the GPU has 4 wrap scheduler per SMP.
```
<img width="1190" height="666" alt="image" src="https://github.com/user-attachments/assets/766df967-378e-44a4-b635-1ca7db373348" />


```
1 Wrap Slot(Number of threads per Warp or Warp size: 32)
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
will use 1 SM and 2 wrap (32 threads each)
```
<img width="1190" height="946" alt="image" src="https://github.com/user-attachments/assets/f49154c2-41ed-46f9-932c-a0485fbe6eca" />

