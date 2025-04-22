## Disjoint Forest

#### Union by rank
$\Rightarrow$ Root of the tree with fewer nodes point to the root of the tree with more nodes.
### Rank : Upper bound on the height of the node
$\rightarrow$ root with smaller rank point to the root with larger rank during a Union operation

*Psuedo Code*
```
MAKE-SET(x)
	x.parent = x;
	x.rank = 0;
UNION(x, y)
	LINK(Find-SET(x), Find-SET(y))
LINK(x, y)
	if(x.rank > y.rank)
		y.parent = x;
	else
		x.parent = y;
		if(x.rank == y.rank)
			// if the height of x and that of y is equal
			// increase 1 to the height of y
			y.rank = y.rank + 1
Find-SET(x)
	if(x != x.parent)
		x.parent = Find-SET(x.parent)
	return x.parent;
```

#### Path Compression
$\rightarrow$ Doesn't change any ranks




#### Maximum Degree
$D(n)$ denote the maximum degree of a node


Creating Empty Heap
$\rightarrow$ Initialize an arrays of root nodes.
$O(1)$

Inserting a new value

Querying the minimum
$O(1)$ 
No change in potential

Uniting two heaps
* Unite the two lists of roots
* Determine the new minimum root
* O(1) amortized time

Deleting the minimum
* Make its children new roots
* Consolidating
	* none of two root node has same degree