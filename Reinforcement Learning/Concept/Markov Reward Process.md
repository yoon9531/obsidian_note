$$
	\left<S, p(s_{t+1} \mid s_t), r(s_t), \gamma \right>
$$
- $S$ : a finite set of states
- $p(s_{t+1} \mid s_t)$ : state transition probability, $s_t$ 에서 $s_{t+1}$ 로 이동할 확률
- $r(s_t)$ : reward function
- $\gamma$ : discount factor (할인율), $\gamma \in [{0, 1}]$
	- 현재의 보상이 미래의 보상보다 얼마나 중요한 지를 나타내는 지표이다.
	- 즉각적으로 얻는 reward와 미래의 얻을 수 있는 reward 간의 중요도를 조절하는 변수
---

### 🧠 Description
- Markov property를 만족하는 상태 전이 과정이다. 
- Agent의 의사 결정 과정이 포함되지 않는다. 즉, 환경의 상태 전이와 보상만 존재하게 된다. -> Policy 없음
- 보상의 누적 합을 최대화 하는 것이 목표임은 마찬가지이다.

즉, MRP는 에이전트가 없어 의사결정이 불가한 Process이다.