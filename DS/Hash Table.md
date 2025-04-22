![[Pasted image 20230517165847.png | 150]]
## Directed-address table
$\Rightarrow$ Hash table that use the ID itself as the index within the array
$\Rightarrow$ NO *collision* $\because$ the number of key == the number of bucket

## Hash table
: directed-address table indexed by *hash values* indexed by *hash values* is called a *hash table*

![[Pasted image 20230517170354.png | 320]]

## Dealing with **collision** 

### *collision*
$\rightarrow$ When more than one key map to the same hash value
$\rightarrow$ $cf. overflow$ (when multiple items are stored in each $bucket$ and it exceeds the size of the bucket)
![[Pasted image 20230518000251.png | 450]]

### Open addrressing
: If an entry is occupied, then you're going to probe another location, see if that location is available

#### Linear Probing
$\Rightarrow$ The next entry in the table is probed
$\Rightarrow$ The $(i+1)^{st}$ probe of key $k$ is made at $(h(k) + i) \ mod \ m$ where $h$ is the hash function and $m$ is the hash table size. 
$\Rightarrow$ This is basically to treat the hash table as a circular structure
##### problem occured
$\rightarrow$ more data or more collison then, more number of data items that all have the same hash value, then they're going to be stored in consecutive block of memory
$\Rightarrow$ **Primary Clustering** : a phenomenon that causes performance degradation in linear-probing hash tables. they have tendency to cluster together into long runs (i.e., long contiguous regions of the hash table that contain no free slots)

#### Qudratic probing
$\Rightarrow$ The $(i+1)^{st}$ probe of key $k$ is made at $(h(k) + c_1 i + c_2 i^2) \ mod \ m$, where $c_1 \ and \ c_2 \neq 0$ are some constants
$\Rightarrow$ Secondary clustering

#### Double hashing
$\Rightarrow$ The $(i+1)^{st}$ probe of key $k$ is made at $(h_1(k) + ih_2(k)) \ mod \ m$, where $h_1 \ and \ h_2$ are (different) hash functions.
$\Rightarrow$ What if $h_2(k)= 500 \ and \ m =1000$? : there going to be *only two distinct locations*
$\Rightarrow$ when is $\left\{ia \ mod \ b \ | \ i \in \mathbb{N} \right\} = \left\{0, \dots, b-1\right\}$? => $a,\ b \rightarrow relatively prime$ 


#### Chaining
$\Rightarrow$ Each entry of the hash table is a linked list
$\Rightarrow$ The number of bins can be smaller than the number of data

### Hash function

#### String as a number
$\Rightarrow$ Characters are internally represented as an integer
ex) ASCII, Unicode, etc.

#### Division
$\rightarrow$ Come up with good divisor and then you return the remainder as the hash value.
Good choice?
> Power of 2?
> Some other number? 303
> Prime number?
> Tend to be bad if the divisor divides $r^k \pm a$ for some small $k \ \& \ a$ 

#### Mid-square
$\Rightarrow$ Take an appropriate the number of bits from the middle of the square
키 값을 제곱한 후 결과 값의 중간 부분에 있는 몇 비트만을 선택하여 버킷주소로 사용.
![[Pasted image 20230522233857.png | 200]]

#### Folding
$\Rightarrow$ Split into several parts of the same length and add them ($shift \ folding$)
$\Rightarrow$ Sometimes reversing every other part($folding \ at \ the \ boundaries$)
![[Pasted image 20230523003612.png | 300]]

#### Digit analysis
* When the keys are known in advance
* Using some radix, examine each digit
* Digits with most skewed distributions are deleted

#### Universal Hashing
> m : hash table size
> For $x \neq y, Pr[h(x)=h(y)]={1 \over m}$ 

* Choose prime $m$ (achieve better distribution of hash values)
* Choose a *piece size*(radix or base) < m
	* piece size determines how many bits are taken at a time from the key to from each $x_i$ in the hashing process
* Choose $a_0,\dots , a_k$(*coefficient*), independently and uniformly at random from $\left\{0,\dots , m-1 \right\}$ 
* Decompose the key $x$ into $x_0, \dots, x_k$ 
* $h(x) = (\sum_{i=0}^k a_ix_i) \ mod \ m$ 


#### Deletion
$\rightarrow$ "deleted" flag

### Load factor

The larger load factor is, more likely to have larger cluster -> degrading the performance
> The number of key / The size of hash table.