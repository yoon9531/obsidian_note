## Hardware Specification
- *Architecture* : x86_64
- *Thread(s) per core* : 2
- *CPU(s)* : 128
- *L1d* : 2 MiB (64 instances) : $2^{21}B$ 
- *L1i* : 2 MiB (64 instances)
- *L2* : 32 MiB (64 instances)
- *L3* : 256 MiB (16 instances)
- *Cache line size* : 64B
``` bash
$ cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
```
## Requirement
1. How did you implement your parallel algorithm 
2. Which approach is better 
3. Reason Why Freivalds’ algorithm is probabilistic and its error bound 
4. Reasons why your program is performing at its best and proving whether performance margin is there or not 
5. Evaluation including tables or graphs 
6. Something else

GEMV (General Matrix-Vector Multiplication) : multiplies a matrix by vector
GEMM (General Matrix Multiplication) : multiplies two matrices

#### Frievlad's algorithm
행렬 곱셈 결과가 맞는지 빠르고 효율적으로 검증할 수 있는 확률적 알고리즘이다.
- 알고리즘 순서
1. Generate random vector $v \in \{0, 1\}^n$ 
2. $p=B\cdot v$
3. $q= A \cdot p$
4. $r = C \cdot v$
5. $q==r$ 이면 패스, 아님 오류

`buf = mat_b X my_vec`
`gemv_out1 = mat_a X buf`
`gemv_out2 = mat_c X my_vec`

mat_a : $N \times N$
mat_b : $N \times N$ 
mat_c : $N \times N$ 
my_vec : $N \times 1$ (consider as row-wise)
buf : $N \times 1$ 
gemv_out1 : $N \times 1$
gemv_out2 : $N\times 1$ 
Flat profile:


성능 결과
GEMM
- naive : 52.48
- multithreading : 2.8658
- 1-level blocking(block size : 64) : 1.567
- 2-level blocking(L1 block: 64, L2 block: 512) : 1.43437
- loop unrolling : 0.86999
GEMV 
- naive: 0.043
- multithreading : 0.01575
- loop unrolling: 0.0061134
