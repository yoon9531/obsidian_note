* Given a weighted (di)graph where all edge weights are *nonnegative*, along with its two vertices s and t, find a shortest path from s to t
* For $v \notin S$, d[v] is the shortest length of an s-v path whose $2^{nd}-to-last$ vertext is in S; prev[v] is the $2^{nd}-to-last$ vertex 
```
Given G=(V, A) with c:A->Q+ and s,t in V
S : empty d[start]<-0, d[v]<-infinity for all v!=s
while s!==v
	choose x not in S that minimize d[x]
	add x to S
	for each <x, v> in A such that v not in S
		d[v] = min{d[v], d[x]+c(x,v)} ; update prev accordingly
return d[t]; shortest length from s to t
```

```
DIJKSTRA(G, w, s)
	Initialize-single-source(G, s)
	S<-empty
	Q = G.V
	while !Q.isEmpty()
		u = EXTRACT-MIN(Q)
		S = S + {u}
		for each vertext v in G.Adj[u]
			RELAX(u, v, w)
```

#### Running time
$\rightarrow$ $O(|V|^2)$ 

### Retrieve
> In addition to length of shortest paths, the paths themselves can be retrieved by tracking back prev[] 