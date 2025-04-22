### Actor-critic Algorithm
우리는 지금까지 누적 보상 합의 기댓값 $J(\theta)$의 gradient를 구하고 policy gradient를 통해 정책을 학습하게 했다. 이때 다른 조치 없이 **vanilla policy gradient**만 사용하는 경우, 추정치의 분산이 매우 크다는 문제가 발생했다.  
따라서 우리는 이 분산을 줄이기 위해 다음과 같은 방식들을 도입하였다.

1. **Reward-to-go**  
   전체 trajectory의 reward 합계 대신, 특정 시점 이후의 미래 reward만을 합산하여 사용한다:  
   $$\hat{Q}_t = \sum_{t'=t}^T r(s_{t'}, a_{t'})$$  
   이는 과거의 불필요한 보상을 제거하여 분산을 줄이는 데 기여한다.

2. **Baseline 기법**  
   추정치에 편향을 주지 않으면서 분산을 줄이기 위해 baseline을 도입할 수 있다. 다음과 같은 형태로 쓸 수 있다:  
   $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \sum_t \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i) \left( R_t^i - b(s_t^i) \right)$$  
   여기서 $b(s_t)$는 상태에 따라 결정되는 baseline 함수이며, 일반적으로 상태 가치 함수 $V^\pi(s_t)$를 사용한다.

3. **Advantage 함수 도입**  
   보상의 기대값과 실제 보상의 차이를 계산하여 정책 업데이트에 반영함으로써 더 정밀한 추정이 가능하다:  
   $$A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$$

4. **Actor-Critic Algorithm**  
   위의 방식들을 종합하면 다음과 같은 구조의 학습 방법이 도출된다:  
   - **Actor**는 정책 네트워크로서 $\pi_\theta(a \mid s)$를 학습하며,  
   - **Critic**은 가치 함수 네트워크로서 $V^\pi(s)$ 혹은 $Q^\pi(s, a)$를 학습한다.  
   - Critic은 TD-Error를 최소화하면서 value function($V, Q$ )을 학습하고, Actor는 Critic으로부터 얻은 Advantage를 바탕으로 정책을 업데이트한다. 이는 trajectory를 진행하는 전체 step 동안 정해진 주기 마다 업데이트 된다.

이와 같은 방식은 REINFORCE에 비해 **매 timestep마다 업데이트가 가능**하고, 추정치의 분산이 낮기 때문에 샘플 효율성(Sample Efficiency)이 높다는 장점이 있다. PPO, A2C, A3C 등의 여러 최신 알고리즘들도 이 Actor-Critic 구조를 기반으로 하고 있다
하지만 이 Actor-Critic 접근에도 문제가 있다. 
1. *Advantage estimation이 정확하지 않을 수 있다*. 실제로는 $Q^\pi$와 $V^\pi$ 모두 정확한 값을 알 수 없기 때문에 Critic이 이를 학습하고 추정해야한다. 즉 value function이 정확하지 않으면 Advantage가 부정확하게 추정되고 학습 초반에는 오차가 매우 커서 정책이 잘못된 방향으로 업데이트 되어 Actor가 최적이 아닌 정책으로 수렴할 수 있다 $\rightarrow$ **N-step returns**
2. 이 *기본 Actor-Critic은 데이터를 한번 수집하고, 한 번만 policy update에 사용하고 버리게 된다*. 이는 수집한 데이터에서 더 많은 정보를 뽑아낼 수 있음에도 사용하지 못해 낭비되는 문제가 있으며 이는 sample efficiency가 낮다는 것을 의미한다. 따라서 데이터를 더 잘 활용하면 더 효율적으로 학습할 수 있게된다. $\rightarrow$ **Importance Sampling**
#### N-step Returns
Q-value 추정 방식에는 다음 3가지가 있다.
1. Monte Carlo : trajectory의 총 return을 사용한다. 하지만 하나의 trajectory로만 계산하기 때문에 결과가 불안정하다는 단점이 있다.
	$$
	\hat Q_{MC}^\pi(s_t, a_t) = \sum_{t^\prime=t}^Tr(s_{t^\prime}, a_{t^\prime})
	$$
2. 1-step TD : 현재 value function으로 추정하므로 안정적이지만 $V^\pi$가 잘못된 추정이라면 $Q^\pi$도 틀려진다는 단점이 이 있다.
	$$
	\hat Q^\pi(s_t, a_t)=r(s_t, a_t) + \gamma V^\pi(s_{t+1})
	$$
3. 위 두 방식을 타협한 것이 N-step return 방식이다. 이 방식은 적절한 n을 선택하는 것이 관건이다. 만약 n이 작다면 variance가 낮아지고 bias가 커진다. 반대로, n이 커지면 variance가 커지고 bias가 작아진다.
	![[Pasted image 20250414005256.png | 200]]
	이 그림을 바탕으로 설명하면 위로 갈수록 Monte Carlo 방식(high variance, many rewards)에 가깝고 아래로 갈 수록 1-step TD 방식(low variance, high bias)에 가깝다.
	$$
	\hat Q_n^\pi(s_t, a_t)=\sum_{t^{\prime}=t}^{t+n-1}r(s_{t^\prime}, a_{t^\prime}) + \gamma^n V^\pi(s_{t+n})
	$$
	하지만 이 N-step return 방식에도 n 값을 고정하기 어렵다는 문제가 있다. 이 문제를 해결하기 위해 등장한 것이 GAE(Generalized Advantage Estimation)이다. 
#### GAE(Generalized Advantage Estimation)
이 방법은 여러 n-step advantage를 지수적으로 가중합해서 계산하게 된다.
기존 n-step advantage는 다음과 같이 계산된다
$$
\hat{A}_n^\pi(s_t, a_t) = \sum_{t'=t}^{t+n-1} r(s_{t'}, a_{t'}) + V^\pi(s_{t+n}) - V^\pi(s_t)
$$
여기서 n값에 따라 variance와 bias가 달라지는 문제를 해결하기 위해 적당한 n값을 찾아야 하는데 이는 고정값으로 두기 어려우므로 그 절충안으로 GAE가 나온 것이다. 즉, _여러 n값의 결과_ 를 **지수 가중치 평균**해서 advantage를 추정하자는 것이다. 
$$
\hat{A}_{GAE}^\pi(s_t, a_t) = \sum_{n=1}^{\infty} w_n \hat{A}_n^\pi(s_t, a_t), \quad w_n \propto \lambda^{n-1}
$$
여기서 $\lambda \in [0,1]$ 은 hyperparameter로 $\lambda$가 0이면 단기 보상 중심, $\lambda$가 1이면 장기 보상 중심의 추정이 된다. 일반적으로 $\lambda=0.95$가 잘 작동한다. 이를 수식으로 살펴보면 다음과 같다.
- $n=0$인 경우
$$
\hat{A}_{GAE}^\pi(s_t, a_t) = r_t + V^\pi(s_{t+1}) - V^\pi(s_t)

$$
- $n=1$인 경우
$$
\hat{A}_{GAE}^\pi(s_t, a_t) = \sum_{t'=t}^{\infty} r_{t'} - V^\pi(s_t)
$$
또 다른 문제점으로 데이터를 한번 수집한 후, policy update에 사용하고 다시 사용하지 않아 데이터 효율성이 떨어지는 것이 있었다. 즉, 아래 수식에서 
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \left( \sum_{t=1}^{T} r(s_t, a_t) \right) \right]
$$
기댓값이 정책 $\pi_\theta$에 의존하기 때문에, policy gradient를 정확하게 계산하기 위해선 직접 샘플링한 trajectory가 필요하다. 즉, on-policy 데이터를 매번 새로 수집해야한다. 또, update하고 난 뒤에는 $\theta$가 바뀌기 때문에 이전 데이터는 더 이상 쓸 수 없게 된다. 이를 극복하기 위해 나온 것이 important sampling이다. 
이를 알아보기 전에 다음 수식을 보면.
$$
\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{x \sim q(x)}\left[ \frac{p(x)}{q(x)} f(x) \right]
$$
이는 실제로 $p(x)$에서 샘플링 못하고, 대신 다른 분포 $q(x)$에서 샘플링을 할 수 있다고 했을 때 이렇게 수식을 바꿀 수 있다는 것이다. 이를 policy gradient에 적용해보면, 우리가 원래 계산하고 싶었던 것은
$$
\mathbb{E}_{\tau \sim \pi_\theta(\tau)}[f(\tau)]$$
이고, 이것을 이전 정책 $\pi_{\theta_{old}}$ 에서 샘플링한 데이터로 바꾸면 다음과 같다.
$$
\mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}(\tau)}\left[ \frac{\pi_\theta(\tau)}{\pi_{\theta_{\text{old}}}(\tau)} f(\tau) \right]
$$
이를 통해 이전 정책에서 수집한 데이터를 사용할 수 있으며, 이제 trajectory 확률 비$\frac{\pi_\theta(\tau)}{\pi_{\theta_{\text{old}}}(\tau)}$를 알아야 한다. 
이제 우리는 $J(\theta)$의 gradient를 다음과 같이 계산할 수 있게 됐다.
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}(\tau)} \left[ \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_{\theta_{\text{old}}}(\tau)} r(\tau) \right]
$$
여기서 우리는 $\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)$이 트릭을 사용해서 gradient를 다음과 같이 다시 쓸 수 있고,
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}(\tau)} \left[ \frac{\pi_\theta(\tau)}{\pi_{\theta_{\text{old}}}(\tau)} \nabla_\theta \log \pi_\theta(\tau) r(\tau) \right]
$$
Trajectory $\tau$는 다음과 같이 표현될 수 있다는 것을 고려한다.
$$
\pi_\theta(\tau) = \prod_{t=1}^{T} \pi_\theta(a_t | s_t)
$$
이제 이것들을 합쳐주면
$$
\mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}(\tau)}\left[\left( \frac{\prod_{t=1}^{T} \pi_\theta(a_t | s_t)}{\prod_{t=1}^{T} \pi_{\theta_{\text{old}}}(a_t | s_t)} \right) \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) r(\tau)\right]
$$
이 된다. 하지만 이 식은 높은 variance가 나타나기 때문에 
$$
\approx \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}(\tau)}\left[\sum_{t=1}^{T} \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \nabla_\theta \log \pi_\theta(a_t | s_t) r(\tau)\right]
$$ 다음과 같이 근사해서 사용한다. 이 식은 $\pi_\theta \approx \pi_{\theta_{old}}$일 때 잘 근사된다
#### Actor-critic with important sampling
최종적으로 important sampling과 GAE 기법을 적용하여 gradient를 적용하는 과정은 다음과 같다.
1. 샘플 수집
	- 현재 정책 $\pi_\theta$를 기반으로 환경에서 N개의 상태-행동-보상 샘플을 수집한다.
	- 이 샘플들은 현재 정책 $\pi_\theta$에서 수집된 on-policy 데이터이다.
2. Critic 업데이트 및 Advantage 계산
	- Value function $V_\phi^{pi_\theta}(s_t)$를 TD 방법 등으로 업데이트
	- Advantage 계산
3. 정책 파라미터 백업
	$θ_{old} ← θ, θ^{(1)} ← θ$
	- 현재 정책 $\theta$를 백업하여 $\theta_{old}$로 저장한다.
4. Important sampling 기반 gradient로 여러번 Actor 업데이트
	$$
	\nabla_{\theta^{(k)}} J(\theta^{(k)}) \approx \frac{1}{N} \sum_{i=1}^N
	\frac{\pi_{\theta^{(k)}}(a_i | s_i)}{\pi_{\theta_{\text{old}}}(a_i | s_i)}
	\nabla_{\theta} \log \pi_{\theta^{(k)}}(a_i | s_i)
	A_\phi^{\pi_\theta}(s_i, a_i)
	$$
	이전에 수집한 샘플들을 사용해서 정책을 여러 번 업데이트 할 수 있게 됐으며. ratio를 통해 off-policy 정도를 보정한다. 이후 gradient ascent를 수행하여 parameter를 업데이트한다.
	즉$$
	\theta^{(k+1)} \leftarrow \theta^{(k)} + \alpha \nabla_{\theta^{(k)}} J(\theta^{(k)})
	$$
5. 최종 정책으로 갱신
	$$θ ← θ^{(K+1)}$$
이 Important sampling 기법에도 역시 문제가 있다. 여러 번 $\pi_\theta$ 업데이트를 반복하게 되면 $\pi_\theta(a_t \mid s_t)$가 reward가 큰 행동 $a_t$에 과도하게 집중하게 된다. 그 결과로 importance ratio가 매우 커지거나 작아지게 되고, 이로 인해 학습이 불안정해지고 variance가 커지게 된다.
이를 해결하기 위해 정책 간 변화의 폭을 제한하기로 한다. 즉, 업데이트하는 정책이 이전 정책에 비해 너무 멀어지지 않도록 하는 것이다. 그 방법에는 다음 3가지가 있다.
#### 1. 파라미터 공간에서 제한
$$\|\theta - \theta_{\text{old}}\|^2 \leq \epsilon$$
parameter의 L2 norm 차이를 제한한다.
#### 2. TRPO(True Region Policy Optimization) : 정책 분포간 KL 거리 제한
$$ D_{KL}(\pi_\theta \,||\, \pi_{\theta_{\text{old}}}) \leq \epsilon$$
이는 정책 자체가 바뀐 정도를 측정한다. 즉 policy간의 KL Divergence로 변화를 제한한다.
#### 3. PPO(Proximal Policy Optimization): Ratio clip
Importance ratio가 $1 \pm \epsilon$ 범위를 넘지 않도록 clip한다.
$$1 - \epsilon \leq \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \leq 1 + \epsilon$$
## PPO(Proximal Policy Optimization)
이전에 진행한 importance sampling을 사용해 구한 policy gradient는 다음과 같았다.
$$
J(\theta) \approx \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^{T} \text{ratio}_t(\theta) \cdot \hat{A}^{\pi_{\theta}}_{\text{GAE}}(s_t, a_t) \right]
$$
이때, ratio가 너무 커지면 update가 폭주하게 되고 너무 작으면 update가 안되게 된다. 따라서 정책이 이전 정책과 너무 멀어지는 것을 방지해야 한다. 이를 방지하기 위한 방법 중에 하나가 바로 이 PPO이다.
PPO는 Clipping을 통해 극단적인 ratio 값이 나오는 것을 제한하는 방식이다.
1. $\hat{A}^{\pi_{\theta}}_{\text{GAE}} > 0$
	- 이득을 주는 방향으로 계속 ratio를 키우면 $J(\theta)$ 엄청 커질 수 있다. 이를 방지하기 위해 상한을 두는 방식이다.
	$$
	\min(\text{ratio}(\theta), 1+\epsilon)\cdot\hat{A}^{\pi_{\theta}}_{\text{GAE}}
	$$
2. $\hat{A}^{\pi_{\theta}}_{\text{GAE}} < 0$
	- 손해를 주는 action을 줄여야하는데 ratio를 무한정 줄이게 되면 업데이트가 폭주할 수 있는 문제가 있다.
	- 따라서, 줄일 수 있는 값의 하한을 둔다.
	$$
	\max(\text{ratio}(\theta), 1-\epsilon)\cdot \hat{A}^{\pi_{\theta}}_{\text{GAE}}
	$$
이를 통해 최종적으로 도출할 수 있는 $J(\theta)$는 다음과 같다.
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^{T} \min \left( \text{ratio}_t(\theta) \cdot \hat{A}_t, \ \text{clip}(\text{ratio}_t(\theta), 1 - \epsilon, 1 + \epsilon) \cdot \hat{A}^{\pi_{\theta}}_{\text{GAE}} \right) \right]

$$
이를 사용하면 Advantage를 기준으로 정책이 너무 빨리 변하는 것을 방지해준다. 특히 특정 action을 지나치게 선호하거나 회피하지 않도록 적절하게 균형을 맞춰줄 수 있다.
따라서 PPO의 전체 학습 흐름은 다음과 같다.
1. 샘플 수집
	- 현재 policy $\pi_\theta$를 사용해 N-step 데이터를 수집한다.
	- $\{(s_i, a_i, r_i) \}_{i=1}^N \sim \pi_\theta$ 
2. Critic 업데이트 및 Advantage 계산
	- Value network $V_\phi^{\pi_\theta}(s)$를 업데이트한다.
	- Advantage 추정: $\hat{A}^{\pi_{\theta}}_{\text{GAE}}(s,a)$
3. 정책 파라미터 초기화
	- $\theta_{old} \leftarrow \theta$
	- $\theta^{(1)} \leftarrow \theta$
4. K번 정책 반복 업데이트
	$$
	J(\theta^{(k)}) = \frac{1}{N} \sum_{i=1}^N \min \left( r_i(\theta^{(k)}) \hat{A}^{\pi_\theta}_{GAE},\ \text{clip}(r_i(\theta^{(k)}), 1 - \epsilon, 1 + \epsilon) \hat{A}^{\pi_\theta}_{GAE} \right) + H(\pi_{\theta^{(k)}})
	$$
	이 정책에 대해 policy를 다음과 같이 업데이트 한다.
	일5. 최종 업데이트	$$
	θ ← θ^{(K+1)}
	$$
왜 PPO 방식이 좋을까?
- 구현 방식이 간단하다
	- 이전의 TRPO(Trust Region Policy Optimization)는 복잡한 제약 조건 최적화를 풀어야 했지만,  PPO는 그걸 간단하게 클리핑(clipping)이라는 방식으로 대체했다.
	- 복잡한 수식이나 2차 도함수 없이, **gradient descent로만 학습 가능**하기 때문에 구현이 쉽다.
- Discrete/continuous action 둘다 지원
	- policy gradient 기반 알고리즘이기 때문에 액션의 형태에 크게 구애받지 않는다.
- Scalable and Parallelizable
	- PPO는 **rollout 환경에서 병렬로 수집**할 수 있다.
- Stable updates
	- TRPO처럼 복잡하게 true region을 직접 계산하지 않고도 clipping을 통해 안전한 범위 내에서만 업데이트할 수 있도록 해준다. 따라서, 학습 안정성이 증가하게 된다.
하지만 rollout length($N$), batch size, entropy coefficient, learning rate 등의 hyperparameter에 민감하다는 단점이 있다. 또, 아직, ppo에서도 on-policy 데이터를 사용한다. ($\frac{\pi_\theta}{\pi_{\theta_{old}}})$ importance sampling과 clipping때문에 off-policy 데이터도 사용하고, policy 데이터가 이전 policy에 비해 너무 커지는 긍정적인 효과도 있었지만 여러 개의 다른 policy들의 정책을 섞어 쓰기 어렵다.
