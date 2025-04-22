## 📖 Definition

Markov Decision Process는 다음 tuple로 정의할 수 있다.
$$
\left<S,  A,p(s_0), p(s_{t+1} \mid s_t, a_t), r(s_t, a_t), \gamma\right>
$$
MRP와의 차이라면, 에이전트의 존재 유무이다. 보상의 누적합을 최대로 한다는 목적은 같지만 Agent의 의사결정을 통한 행동이 추가된다. 이는 위 튜플에서 $A$와 $a_t$가 추가 된 것으로 확인할 수 있다.
- $S$ : set of states
- $A$ : set of actions
- $p(s_0)$ : Initial state distribution
- $p(s_{t+1} \mid s_t, a_t)$ : state transition 확률
- $r(s_t, a_t)$ : reward function, 행동 $a_t$ 를 취한 후 상태 $s_t$ 에서 얻는 보상
- $\gamma$ : discount factor

## 📝 Key Concepts

- Markov Property전
	- Future state는 현재 state에 의해서만 결정된다. 
	- = Future state는 과거 state에 독립적이다.
	- = state는 기록에 남겨져 있는 모든 관련 정보를 저장한다.
	- If the state is **Markov** if and only if
$$
p(s_{t+1} \mid s_t) = p(s_{t+1} \mid s_1, \dots , s_t)
$$
- 전제 조건은 모든 state가 Markov property를 따른다는 것이다
### 한 사이클의 흐름 
- 에이전트는 상태 $s_t$​를 관찰
    
- 정책 $\pi(a_t | s_t)$에 따라 행동 $a_t$​ 선택
    
- 환경은 다음 상태 $s_{t+1}$과 보상 $r_t$를 반환
    
- 에이전트는 이를 바탕으로 학습
    
- 이 과정을 반복하여 누적 보상 합이 최대가 되도록한다.
## 🗃️ Related Topics

- [[Markov Reward Process]]
    
- [[Markov Property]]    

## ❓ Common Questions

## 🧠 Summary in One Sentence

## 📅 Dates

- First studied: 2025-04-01
    
- Last reviewed:
    

## 🏷️ Tags

#Senior #Markov #RL 