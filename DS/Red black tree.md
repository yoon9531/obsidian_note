$\Rightarrow$ *Self-balancing* binary search tree data structure
$\Rightarrow$ Ensures that the tree always has a logarithmic height

For notational simplicity we will treat the 'null' pointers as leaf nodes.

* root nodes always black
* null pointer considered as leaf nodes with black color
* the children of red nodes must be black
* Consider all paths to the null pointer, the the number of black nodes visited thorugh the path is always same

A red-black tree with a () nodes has height $<= 2log_2(n+1)-1$ 

#### Rotations
* BST properties are presented
* prevent the tree grows in the certain specific direction

#### Insertion
* Insert node with color red on usual operation
* if the parent is red or if the new nodes is the root node, two properties can be violated

#### Deletion
* Implementation regular BST deletion first
* Let *x* be the child of disappearing node replacing it
* If the *disappearing node* is red, done
* If we could place an **extra black** on x we would be done
* Delete in usual way(regular BST deletion)
* A while loop where x is the node with extra black
~~~
x <- node with an "extra black"
// check whether the x is already black (checking doubly black)
while x is not root and color is black
	(assume that x is the left child of its parent)
	// case 1
	// After the operation, we always have red parent
	// results in red parent and leads to one of Cases 2-4
	if the sibling is red
		color the sibling black
		color the parent red
		left-rotate at parent
	else
		// sibling is black
		// case 2
		// does not change the tree topology and x goes up
		// If the parent was red, leads to termination
		// only case occured in the number of multiple times
		if both "nephews" (children of sibling)are black
			color the sibling red
			the parent becomes x
		else
			// case 3 always lead to case 4
			if only the "right nephew" is black
				color the "left nephew" black
				color the sibling red
				right-rotate at the sibling
			// case 4 always leads to termination of while loop
			swap the colors of the sibling and the parent
			color the "right nephew" black
			left-rotate at the parent
			root node becomes x
color x black
~~~
