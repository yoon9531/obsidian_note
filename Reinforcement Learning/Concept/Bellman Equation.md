#### 어떤 상태에서의 가치가 현재 보상과 미래 가치의 합이라는 것을 수학적으로 표현한 것

### State-value function
상태 가치 함수 $V^{\pi}(s)$의 Bellman equation은 다음과 같이 표현할 수 있다.
$$
V_{\phi}^{\pi_\theta}(s_t)=E_{a_t\sim \pi(a_t, s_t), \,s_{t+1}\sim p(\cdot\, \mid s_t,a_t))}\left[r_t+V^{\pi}(s_{t+1})\right]
$$
여기서 $V^{\pi_\theta}_\phi(s_t)$ 는 상태 $s_t$에서 정책 $\pi$를 따랐을 때의 기대 누적 보상이다. 즉, 전체가지 = "현재 보상" + 미래 가치를 수학적으로 표현한 식이다.
### State-action value function
상태 행동 가치 함수에서는 가치 함수에서 현재 상태에서의 행동까지 고려하여 그 보상을 계산한다. 즉, 현재 상태 $s_t$에서 행동 $a_t$를 했을 때의 기대 return 값을 나타낸 식이다.
$$
Q_\phi^{\pi_\theta}(r_t, s_t) = \mathbb{E}_{s_{t+1} \sim p(\cdot \mid s_t, a_t)} 
\left[ r_t + \mathbb{E}_{a_{t+1} \sim \pi(\cdot \mid s_{t+1})} 
\left[ Q^\pi(s_{t+1}, a_{t+1}) \right] \right]
$$