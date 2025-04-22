- Given a graph G=(V, E), we say a vertex v is an <strong>articulation point</strong> (or a **cut vertex**) if its removal disconnects G
- we say a graph is *biconnedted* if it has no articulation points
- A *biconnected component* of a connected graph G is a **maximal**(w.r.t vertex set inclusion) biconnected subgraph of G 
- Every edge has to belong to exactly one biconnected component
![[biconnectedGraph.png|500]]
### Maximal
: state that no more vertexes can be added.

## Articulation Point in DFS Tree
[[Depth First Search(DFS)]]
![[Pasted image 20230515230525.png | 200]]

- $dfn[v]$ = i if v was the i-th vertex visited by DFS
- $low[v]$ is the smallest $dfn$ of the endpoints of the back edges incident to v's descendants(or $dfn[v]$ if that is smaller)
- $low[v]$ = min {$dfn[v]$,  $low[x]$ : x is child of v, $dfn[x]$ : (v, x) is a back edge}
- A non-root node v is a cut-vertex iff it has a child x s.t. $low[x] \geqq dfn[v]$
	- Let $low[u] \geqq dfn[v]$, then ancestor of v has smaller $dfn$ than $dfn[v]$. but $low[v]  \geqq dfn[v]$ it means that there is are no back edges from u to v's ancestor. so $v$ is articulation point
- The root node is a cut vertex iff it has more than 2 children

## Find Articulation Point
1. root node : check if it has more than 2 children
2. non-root node
	1.  Run DFS and keep updating $low$ value
	2.  If explore is over, comparing $dfn[v]$, $low[u]$ , if $low[u] \geqq dfn[v]$, $v$ is a *Articulation Point*

~~~
DFS(v, p = 0 // parent node) {
	visited(v) = true;
	dfn[v] = low[v] = counter++;
	children = 0; // Number of children

	for each vertex u adjacent to v {
		// if the vertex is already visited, it has a back edge (u, v)
		if (u is visited) {
			low[v] = min(low[v], dfn[u]);
		}
		// if u is not visited
		else {
			DFS(u, v);
			// This code block is performed after all the children
			// of v is visited by DFS
			low[v] = min(low[v], low[u]);
			if(low[u] >= dfn[v]) {
				IS_CUTPOINT[v];
				++children
			}
		}
	}
	if(p == 0 && chilren > 1) {
		IS_CUTPOINT[v];
	}
}
~~~

## Find Biconnected Component
$\rightarrow$ No matter which two biconnected components of given graph you may take, it will never be sharing more than one vertex.
1. explore vertex v, push every edge $(v,u)$ to Stack
2. After or during explore, if it is confrimed that v is articulation point ($low[u] \geqq dfn[v]$), or DFS from root node is over, then pop from the stack until edge $(u,v)$ is popped
3. And the set of popped elements is *Biconnected Component*
~~~
DFS(v) {
	visited[v] = true;
	Stack s;
	dfn[v] = low[v] = counter ++;
	for each vertex u adjacent to v {
		low[v] = min(low[v], dfn[u]);
			push (v, u) to stack
		}
		else {
			push (v, u) to stack
			DFS(u);
			low[v] = min(low[v], low[u])
			if(low[u] >= dfn[v]){
				pop all edges until (v, u) is represented and
				return them as single biconnected component
			}
		}
	}
}
~~~
```
The main rountine calls findBC(root, dummy (root node has no parent)),
with dfn[] initialized as zero
v : arbitrarily chosen vertex
procedure findBC(v, parent)
	// define global counter that counts how many vertices have been
	// visited, increase counter and then set that number as the dfn
	set dfn[v]
	low[v] <- dfn[v]
	for each incident edge(v, x)
		// every edge is going to be considered only one
		// edge is going downwards -> not so interesting
		if x = parent or dfn[v] < dfn[x] then continue
		push(v, x)
		// we can detect whether a vertex has been visited or not by
		// checking if the dfn of vertex is zero or not
		// we are now taking a look of a back edge
		if(x has been visited) then
			// compare dfn[x] and low[v]
			update low[v]
		// I'm taking a look of an interesting edge
		// But it was not previously visited -> i'm looking child
		else
			findBC(x, v)
			// function return -> we know that low of the child has been
			// already computed -> we have chance to update low of v
			// this time by comparing low[v] agains low[x]
			update low[v]
			// identify cut vertex
			// low of sub x is greater than equal to dfn of myself
			// that's the moment that i realized that i'm a cut vertex
			if low[x] >= dfn[v] then
				repeat
					pop and ouput an edge
				until the edge is (v,x)
```