[[Union-Find]]
#kruskal #unionfind #mst #minimum #spanning #tree
$T \leftarrow \varnothing$ 
Consider each edge $e$ of $G$ in nondecreasing order of cost
	1. If $e$ can be added to $T$ without creating a cycle, add it
	2. Stop if $T$ contains $|V|-1$ edges
If $T$ contains fewer than |V|-1 edges, $G$ is disconnected
___
> In each iteration that adds an edge
> 	let $e$ be the edge considered in this iteration and $v$ be its arbitrary endpoint
> 	let $S \subsetneq V$ be the connected component containing $v$.
> 	$e$ has the minimum cost among all the edges btw $S$ and $S \ and \ V-S$.

### Key lemma
 > Lemma 1.
		Given a weighted graph $G=(V,E)$ with distinct edge cost $c\ : \ E \rightarrow  \mathbb{R}$ , let $S \subsetneq V$ be a nonempty set of vertices and $e \in E$ be the **minimum cost edge** among all the edges between $S$ and $V-S$. Then, every minimum spanning tree of $G$ contains $e$.
	
	$Proof$
		Let $u \in S$ and $v\notin S$ be the two endpoints of $e, \ e=(u,v)$. Suppose toward contradiction that there exists a minimum spanning tree $T$ such that $e\notin T$.
	    There exists a unique simple path between $u$ and $v$ in $T$, and this path contains at least one edge between $S$ and $V-S$; let $f$ be such an edge. Since $e$ is the minimum-cost edge among the edges between $S$ and $V-S$, we have $c(e) < c(f)$. We claim that $T^{\prime} := T \cup \left\{e \right\} - f$ is a spanning tree: it is easy to see that $T^{\prime}$ contains exactly $|V|-1$ edges, and $T^{\prime}$ is connected because $f$ is an edge on the unique cycle of $T \cup \left\{e \right\}$ and removing it preserves connectivity. On the other hand, since $c(e) < c(f)$, the total cost of $T^\prime$ is strictly smaller than that of T, contradicting our choice of T.  
___
#### If $|T|=n-1$ at the end of algorithm
* $T$ is acyclic and thus a tree
* every edge in $T$ belongs to every $MST$
* every $MST$ has $n-1$ edges
* $T$ is the (unique) minimum spanning tree
#### If $|T| \neq n-1$ at the end of algorithm
* $T < n-1$ and thus $(V, T)$ is not connected
* $G$ does not have an edge btw any two connected components of $(V,T)$.
* $G$ does not have $MST$.
___
#### Running Time

$O((|V|+|E|) + |E|log|E| + |E|\cdot|V|)$ 
$\rightarrow$ $O(|V|\cdot |E|)$ 
___ 
#### Pseudo Code
```
MST-Kruskal(G, w)
A = ∅
for each vertex v in G.V
	MAKE-SET(v)
sort the edges of G.E into nondecreasing order by weight w
for each edge(u, v) in G.E taken in nondecreasing order by weight
	if(FIND-SET(u) !== FIND-SET(v))
		A = A + {(u, v)}
		UNION(u, v)
return A
```
___
### Sample Code

~~~
package DataStructure.Graph;  
  
import java.io.BufferedReader;  
import java.io.IOException;  
import java.io.InputStreamReader;  
import java.lang.reflect.Array;  
import java.util.*;  
  
class KEdge {  
    int u;  
    int v;  
    int w;  
    KEdge(int u, int v, int w) {  
        this.u = u;  
        this.v = v;  
        this.w = w;  
    }  
}  
  
public class KruskalAlgorithm {  
    static int[] parent;  
    static ArrayList<KEdge> graph;  
    static int V; // the number of vertices  
    static int E; // the number of edges  
    public static void main(String[] args) throws IOException {  
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));  
  
        V = Integer.parseInt(br.readLine());  
        E = Integer.parseInt(br.readLine());  
  
        graph = new ArrayList<>();  
        parent = new int[V];  
  
        for(int i = 0; i < V; i++) {  
            parent[i] = i;  
        }  
  
        for(int i = 0; i < E; i++) {  
            StringTokenizer st = new StringTokenizer(br.readLine());  
            int u = Integer.parseInt(st.nextToken());  
            int v = Integer.parseInt(st.nextToken());  
            int w = Integer.parseInt(st.nextToken());  
  
            graph.add(new KEdge(u, v, w));  
        }  
    }  
  
    public static void Kruskal() {  
        // sorting the edge in ascending order(minimum cost -> maximum cost)  
        graph.sort(new Comparator<KEdge>() {  
            @Override  
            public int compare(KEdge o1, KEdge o2) {  
                return o1.w - o2.w;  
            }  
        });  
  
        for(int i = 0; i < E; i++) {  
            KEdge edge = graph.get(i);  
            if(find(edge.u) != find(edge.v)) {  
                union(edge.u, edge.v);  
            }  
        }  
  
    }  
    public static void union(int a, int b) {  
        a = find(a);  
        b = find(b);  
        if(a < b) parent[b] = a;  
        else parent[a] = b;  
    }  
    public static int find(int x) {  
        if(parent[x] == x) {  
            return x;  
        }  
        return find(parent[x]);  
    }  
  
}
~~~
