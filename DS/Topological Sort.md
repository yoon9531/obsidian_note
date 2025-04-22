$\Rightarrow$ Given a directed graph, find a total ordering of vertices s.t. for every arc <u, v>, u precedes v in the ordering
```
TOPOLOGICAL-SORT(G)
	call DFS(G) to compute finishing times v.f for each vertex v
	as each vertex is finisehd, insert it onto the front of a linked list
	return the linked list of vertices
```

#### Detecting Cycle
```
private boolean dfs(int u, int parent, boolean[] visited) {
	visited[] = true
	for(int v : adj[v].root) {
		if(!visited[v]) {
			if(dfs(v, u, visited)) {
			return true;
			}
		} else if (u != parent) {
			// back edge detected
			return true;
		}
	}
	
	return false
}
```