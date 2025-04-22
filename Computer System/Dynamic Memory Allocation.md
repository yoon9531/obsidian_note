#### Knowing How much to free
1. Standard method
Keep the length of a block in the word preceding the block
**header field or header**
- Method 1 : *Implicit list using length* - links all blocks
- Method 2 : *Explicit list* among the free blocks using pointers
- Method 3 : *Segregated free list*
- Method 4 : *Blocks sorted by size* -> Can use a balanced tree
### Implicit List
- For each block we need both *size* and *allocation status*
- some low-order address bits are always 0
- Instead of storing an always-0bit, use it as a *allocation/free flag*
	- 1 : Allocated , 0 : Free block
- size and flag in *1 word*
#### First fit 
- Search list from beginning, *choose first free block* that fits
- Can take linear time
#### Next fit
- search list starting where previous search finished
#### Best fit
- Choose the best free block: fits with fewest byte left over

#### Allocating in Free block
1. splitting : since allocated space might be smaller than free space, we might want to split the block.
#### Freeing a block
- Need only clear the allocated flag
- But can lead to false fragmentation
#### Coalescing
- Join(coalesce) with next/previous blocks, if they are free
- Coalescing with next block
#### Bidirectional Coalescing
- *Boundary tags*
	- Replicate size/allocated word at "bottom" (end) of free blocks
	- Allows us to traverse the list backwards, but requires extra space
#### Dealing with Memory Bugs
- Debugger : gdb
- Data structure consistency checker
- Binary translator : valgrind
- glibc malloc contains checking code
	- setenv MALLOC_CHECK_3
#### Address Sanitizer
- Detect memory access error
- Performance is acceptable