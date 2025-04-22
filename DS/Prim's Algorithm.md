* Let $T$ be a singleton graph of an arbitrary vertex in G
* Repeat the following |V|-1 times:
	* Let $e=(u,v)$ be the minimum cost edge btw $V(T)$ and $V(G)-V(T)$, where $u\in V(T)$ and $v\in V(G)-V(T)$.
	* If such $e$ does not exist, $G$ is disconnected
	* Add $v$ and $e$ to $T$
#### Runnign time
$\rightarrow$ $O(|V|\cdot (|E|+|V|))$ 
$\rightarrow$ $O(|V|\cdot |E|)$ - a little optimization
___
#### Pseudo Code

```
MST-PRIM(G, w, r)
	for each u in G.V
		u.key = infinity
		u.parent = NIL
	r.key = 0
	Q = G.V
	while Q !== 0
		u = EXTRACT-MIN(Q) // O(VlogV)
		for each v in G.Adj[u] // O(ElogV) 
			if v in Q and w(u, v) < v.key
				v.parent = u
				v.key = w(u.v)
```
Line 8 : O(VlogV)
Line 9 ~ 12 : O(ElogV) : the length of adjacency list  = 2|E|, line 12 involves decrease-key operation on min-heap.
-> O(VlogV + ElogV) = O(ElogV)
*Improvement* -> Using Fibonacci Heap
O(E + VlogV)

___

#### Sample Code
~~~
package DataStructure.Graph;  
  
import java.util.*;  
import java.io.*;  
  
/* Let T be a singleton graph of an arbitrary vertex in G  
     Repeat the following |V|-1 times:      Let e=(u,v) be the minimum cost edge btw V(T) and V(G)\V(T),      where u in V(T) and v in V(G)\V(T)     if such e does not exist, G is disconnected     Add v and e to T*/  
public class PrimsAlgorithm {  
    static ArrayList<ArrayList<Edge>> graph;  
    static int answer = 0;  
    static boolean[] visited;  
    static PriorityQueue<Edge> pq;  
  
    public static void main(String[] args) throws IOException {  
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));  
        StringTokenizer st = new StringTokenizer(br.readLine());  
  
        final int V = Integer.parseInt(st.nextToken()); // Number of vertex  
        final int E = Integer.parseInt(st.nextToken()); // Number of vertex  
  
        visited = new boolean[V+1];  
        graph = new ArrayList<>();  
        pq = new PriorityQueue<>();  
  
        for(int i = 0; i < V+1; i++) {  
            graph.add(new ArrayList<>());  
        }  
  
        // Make graph which is composed by e=(u,v) and cost w(e)  
        for(int i = 0; i < E; i++) {  
            StringTokenizer input = new StringTokenizer(br.readLine());  
            int u = Integer.parseInt(input.nextToken());  
            int v = Integer.parseInt(input.nextToken());  
            int w = Integer.parseInt(input.nextToken());  
  
            graph.get(u).add(new Edge(v, w));  
            graph.get(v).add(new Edge(u, w));  
        }  
  
        // Starting point : vertex 1  
        pq.add(new Edge(1, 0));  
  
        while(!pq.isEmpty()){  
            // At first time, this must contain starting point  
            Edge edge = pq.poll();  
            int v = edge.v;  
            int cost = edge.cost;  
  
            // if v is already visited, we don't need to check again.  
            if(visited[v]) continue;  
  
            visited[v] = true;  
            // answer will be the final MST's total cost  
            answer += cost;  
            // visit all the adjacent vertex from vertex v  
            for(int i = 0; i < graph.get(v).size(); i++) {  
                Edge e = graph.get(v).get(i);  
                // if all the adjacent vertex have not been visited  
                // add it to priority queue                // priority queue will automatically enumerate it in ascending order of edge cost                // then this algorithm cannot help connecting u and v (e=(u,v) be the minimum cost edge btw V(T) and V(G)\V(T))                if(!visited[e.v]) {  
                    pq.add(e);  
                }  
            }  
        }  
  
  
        System.out.println(answer);  
  
    }  
}
~~~

## Prim's Algorithm (Using Heap)
#Heap #minheap #Decrease-key
#### Running time
$\rightarrow$ $O(|V|+|E|log|E|)$
* E|log|E|$  : Heapify every time the key decrease
$\rightarrow$ $O(|E|log|V|)$ *if $G$ is connnected*
