#### Init value
- BATCH ($B$)= 32
- IN_DIM ($\text{in\_dim}$)= 4096
- OUT_DIM ($\text{out\_dim}$)= 4096
- RANK ($r$)= 8
#### Arguments
- Input arguments
	- x ($B, \text{in\_dim}$): input matrix 
		- Size : $32 \times 4096 \times 4 = 2^{19} B$
	- W ($\text{out\_dim}, \text{in\_dim}$): original weight
		- Size : $4096 \times 4096 \times 4 = 2^{26}\,B$
	- A ($r, \text{in\_dim}$}): Lora down projection
		- Size : $8 \times 4096 \times 4 = 2^{17}B$
	- B ($\text{out\_dim}, r$): Lora up projection
		- Size : $4096 \times 8 \times 4 = 2^{17}B$
	- $\alpha(=16.0f)$ : scaling vector
		- Scale = 16.0f / 8(=rank)
- Output ($B, \text{out\_dim}$): y

#### Process
1. `out_linear` 계산
	- $\text{out\_linear} = x \times W^T$
	- $B \times \text{in\_dim} \times \text{in\_dim} \times \text{out\_dim} \rightarrow B\times\text{out\_dim}$
2. `out_lora` 계산
	- $x \times A^T \quad (B \times r)$ 
	- $\left(x\times A^T\right) \times B^T \quad (B\times \text{out\_dim})$
	- $\left(x\times A^T\right) \times B^T \times \frac{\alpha}{r}$
3. `y` 계산
	- `y = out_linear + out_lora`
	- 원소 별로 더하기
#### Hardware Spec
- TMUs : 328
- Memory Size: 24GB
- SM : 82개
- Max warp 수  : 64
- warp당 thread 수 : 32
- 따라서 SM당 2048개 thread handling 가능
- 이러한 SM이 82개
- L1 cache : 128KB (per SM)
- L2 cache : 6MB
- CUDA core 수 : 10496
- SM 개수 : 82
- Max threads per SM = 1536
![[Pasted image 20250425115030.png|500]]
#### 최적화 아이디어
- **타일링**·**공유 메모리**로 A·B 블록 재사용
- **warp coalesing**으로 연속 메모리 접근
- **스트림·비동기 복사**나 **커널 합치기** (fuse) 등을 활용해 오버헤드를 줄이기.

Max threads per block : 1024
Warp size  = 32
SM : 82
Max threads per SM = 1536
Max blocks per SM = 16
SM당 warp수 = 1536 / 32 = 48 warps

한 thread block당 1024개의 thread를 사용할 수 
#### Time measurement
1. Naive 
	- Each threads are responsible for one element of output matrix $y(i, j)$
	- `Lora kernel time : 7.001ms`
2. Tiling + Shared memory
	