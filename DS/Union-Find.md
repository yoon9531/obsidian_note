[[Kruskal Algorithm]]
#union #find #weighteduion #cardinality
## Union Find
* Maintains a family of disjoint subsets of $\{0, \dots, n-1\}$ 
* Supports the following operations
	* Initialization with $n$ singletons
	* $Union(i, j)$ : unites $S_i \ and \ S_j$
	* $Find(x)$ : returns the (name of the) set that contains $x$
___
![[Pasted image 20230518215626.png | 300]]
#### Union
* Make the one tree a subtree of the other's root
* $O(1)$ time
#### Find
* Find the root
* $O(n)$ time
#### Initialization
* $O(n)$ time (worst case)
___ 
## Improvement
$\Rightarrow$ *Weighted union* : make the smaller-cardinality tree a subtree
* cardinality : measure of the number of elements of the set.
* Maximum height of a tree representing a set of cardinality k : $log_2 \ k$ 

![[Pasted image 20230518215705.png | 300]]
#### Union
* Make the one tree a subtree of the other's root
* $O(1)$ time
#### Find
* Find the root
* $O(log \ n)$ time
#### Initialization
* $O(n)$ time

### Path comprehension
all the elements we have seen on the way
Not all find operations can take that much time
```
MAKE-SET(x) 
	x.parent = x
	x.rank = 0
UNION(x, y)
	LINK(FIND-SET(x), FIND-SET(y))
FIND(x) 
	if x != x.parent
		x.parent = FIND(x.parent)
	return x.parent
LINK(x, y)
	if(x.rank > y.rank)
		y.parent = x
	else
		x.parent = y
		if(x.rank == y.rank)
			y.rank++
```


#### Armotized analysis
Analyze the "running time" required to perform on operation in a given data structure

Any mixed sequence of f finds and u unions (u >= n/2) takes O(n+a(f+n,n)) time where a is the inverse Ackermann's function

> Ackermann's function
> A(1, j) = 2i for j>=1
> A(i, 1) = A(i-1, 2) for i>=2
> A(i, j) = A(i-1, A(i, j-1)) for i,j>=2
> a(p, n) = min(z>=1 | A(z, [p/n] > log2n))for p, n >=2