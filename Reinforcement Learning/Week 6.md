## Off-policy RL
먼저 On-policy와 Off-policy의 차이부터 살펴보자
#### On-policy vs Off-policy
- On-policy : 학습에 쓰이는 정책과 데이터를 수집하는 정책이 같은 정책 $\pi$를 뜻한다. 이는 매번 최신 정책으로 데이터를 다시 수집해야 하므로 데이터 재사용성이 떨어진다.
- Off-policy : 학습하는 목표 정책 $\pi$와, 데이터를 수집하는 행동 정책 $\mu$가 서로 다르다. 이는 한번 수집한 데이터를 다시 사용할 수 있어 재사용성이 높다.
$$
Q_\phi(s_i, a_i) \leftarrow r_i + \gamma\max_{a_i^\prime}Q_{\phi^-}(s_i^\prime, a_i^\prime)
$$
이를 기반으로 policy는 다음과 같이 greedy한 방법으로 정의된다.
$$\pi(s) = \arg\max_aQ_\phi(s,a)$$
이때, $\max_{a_i^\prime}Q_{\phi^-}(s_i^\prime, a_i^\prime)$는 모든 action space의 action $a^\prime$에 대해 Q값을 계산해야 하는데, continuous action space에서는 사실상 확인할 action이 무한 개이기 때문에 현실적으로 불가능하다. 또, greedy policy $\pi(s)$를 찾을 때 발생할 수 있는 또 다른 문제는, 최적화 문제이다. continuous action space에서는 Q함수의 최댓값을 찾기 위한 최적화 작업이 필요하며(e.g. gradient descent) 이 방법으로 인해 부정확하거나 느려지는 문제가 발생할 수 있다.
따라서 이러한 문제를 해결하기 위해 DQN은 continuous action space에서는 잘 작동하지 않고 다음과 같은 방법이 사용된다.
- **DDPG (Deep Deterministic Policy Gradient)**
- **TD3 (Twin Delayed DDPG)**
- **SAC (Soft Actor-Critic)**
#### Find greedy policy in continuous actions
앞서 살펴봤듯이 다음 방식으로 policy를 찾는 것은 continuous space에서 적절하지 않다.
$$\pi(s) = \arg\max_aQ_\phi(s,a)$$
따라서 deterministic policy, policy network $\pi_\theta(s)$를 근사시키는 방법을 사용한다.
수식으로 나타내면 다음과 같다.
$$
	\max_aQ_\phi(s,a)\approx Q_\phi(s, \pi_\theta(s))
$$
여기서 $\theta$를 학습시켜 $Q_\phi(s, \pi_\theta(s))$값이 최대가 되도록한다.
### DDPG : Deep Deterministic Policy Gradient
Actor-critic + Replay + Target network 구조로 구현한 알고리즘
목적 : Actor가 내놓은 행동에 대한 Q값 최대화하기
#### Stochastic policy vs Deterministic policy
- Stochastic policy $\pi(s \mid a)$ : action에 대한 확률 분포
- Deterministic policy $\pi(s)$ : action에 대해 하나의 확률만 가지는 분포
#### Policy gradient vs Deterministic policy gradient
$$
\text{PG} : \nabla_\theta J(\theta) = \mathbb{E}_{s, a\sim\pi}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot Q^\pi(s,a)\right]
$$
$$
\text{DPG} : \nabla_\theta J(\theta) = \mathbb{E}_{s_t} \left[\nabla_aQ_\phi(s,a) \mid _{s=s_t, a = \pi_\theta(s_t)}\nabla_\theta\pi_\theta(s)\right]
$$
이런 식으로 parameter를 업데이트해서 최적의 policy $\pi_\theta(s)$를 찾을 수 있다.
여기서 $\nabla_\theta J(\theta)$를 구하기 위해서는 $\pi_\theta(s)$ 뿐만 아니라 $Q_\phi(s,a)$도 필요하다. DDPG에서는 
$$
Q^\mu(s,a) = \mathbb{E}_\mu[R_t \mid s,a]
$$ 를 actor-critic 방식의 parameterized Q-function을 이용해서 구한다. 
DQN에서는 다음과 같이 target network를 업데이트했다.
$$
\phi^- \leftarrow \phi \quad \text{every L steps}
$$
DDPG에서는 더 soft하게 target을 update한다.
$$
\phi^- \leftarrow \rho\phi^- + (1-\rho)\phi \quad \text{every step}
$$
DDPG는 target network를 policy와 value function 모두에 사용한다.
#### DDPG의 목적 함수
$$
J(\theta) = \mathbb{E}_{s_t\sim D}[Q_\phi(s_t, a) \mid a=\pi_\theta(s_t)]
$$
"Actor가 내놓은 행동을 critic이 평가시키고, 그 기대 가치를 극대화." 하고 싶다는 뜻이다.

#### $J(\theta)$의 gradient
$$
\nabla_\theta J(\theta) = \nabla_\theta\mathbb{E}_{s_t}[Q_\phi(s_t, \pi_\theta(s_t))]
\approx \mathbb{E}_{s_t}[\nabla_\theta Q_\phi(s_t, a) \mid a = \pi_\theta(s_t)]
= \mathbb{E}_{s_t} \left[\nabla_a Q_\phi(s,a) \mid_{s=s_t,a=\pi_\theta(s_t)} \nabla_\theta\pi_\theta(s) \mid_{s=s_t}  \right]

$$
이 수식에서 알 수 있는 것은 Critic이 좋다고 평가해주는 방향($\nabla_a Q_\phi(s,a))$ 로 Actor 파라미터를 밀어준다는 것이다. 다른 말로 바꿔 말하면, Actor는 Critic이 평가해준 action-gradient를 통해 더 좋은 행동을 하도록 갱신한다.
#### Soft target update
DQN에서는 메인 네트워크 $\phi$ 를 $L$ step동안 학습하고 이를 target network $\phi^-$에 복사하는 방식으로 target을 hard update한다. 하지만 이는 target network가 갑자기 크게 변해서 loss가 불안정해질 수 있다는 단점이 있다. 이를 개선하기 위해 DDPG에서는 soft target update 방식을 취한다. 이는 다음과 같이 표현할 수 있다.
$$\phi^- \leftarrow \rho\phi^- + (1-\rho)\phi \quad \text{every step}$$
여기서 $\rho$는 0.99와 같이 1에 가까운 값을 사용한다. 이를 통해 target network가 천천히 부드럽게 변할 수 있게 해준다. 이를 통해 학습 안정성이 올라간다. DDPG는 이 soft target update 방법을 critic network 뿐만 아니라 때로 actor network에도 적용해 전체 알고리즘의 학습 안정성을 향상시킨다.
### TD3 : Twin Delayed Deep Deterministic Policy Gradient
위 DDPG 또한 Q 값의 overestimation 문제가 있다.
##### Issue 1: Overestimating target value
target value를 업데이트하는 다음 식에서
$$ y_i \leftarrow r_i + Q_{\phi^-}(s_i^\prime, \pi_{\theta^{-}}(s_i^\prime))$$  noise나 오차로 Q가 실제보다 크게 추정되면 이게 누적돼서 overestimate될 수 있고 이로 인해 학습이 불안정해질 수 있다.
$\rightarrow$ Solution : Clipped Double Q-learning
두 개의 Critic $Q_{\phi_1}, Q_{\phi_2}$를 학습하고 타깃은 더 적은 쪽을 쓰는 방식으로 과대 추정을 방지한다.
$$ y_i \leftarrow r_i + \gamma \min_{j=1,2}Q_{\phi_j^-}(s_i^\prime, \pi_{\theta^-}(s_i^\prime))$$
이를 통해 큰 Q값을 선택할 수 없게 clipping해줄 수 있다.
##### Issue 2 : Deterministic policy can quickly overfit to noisy target Q-function
DDPG의 Actor $\pi_\theta$는 Critic이 높은 값을 주는 행동만 골라내기 때문에 critic 근사 오차가 있으면 정책이 이것에 금방 overfit 해버린다.
$\rightarrow$ Solution
1. Delayed policy update
	- Critic을 여러 번 업데이트(every step) 한 뒤에
	- Actor를 더 느리게(2~3배 주기로) 업데이트하여
	- Critic이 어느 정도 안정화 된 값 위에서 정책을 배우도록 한다.
2. Target policy smoothing
	- Target 행동 계산 시 약간의 노이즈를 섞어 준다.
		$$a^\prime=\pi_{\theta^-}(s^\prime) + \epsilon\quad \epsilon\sim \mathcal{N}(0, \sigma)$$
이를 적용해준 것이 TD3이다. 따라서 이전 DDPG에서 target value 업데이트하던 것을 Clipped double Q-learning을 적용한 것으로 바꿔주고, target action을 target policy smoothing을 적용해 주어 다음 식을 추가해준다.
$$
a^\prime(s^\prime)= \text{clip}(\pi_{\theta^-}(s^\prime) + \text{clip}(\epsilon, -c, c), a_{Low}, a_{High}) \quad \epsilon \sim \mathcal{N}(0, \sigma)
$$
또, 매번 policy를 업데이트 해주던 것도 지정한 delay값 만큼 기다린 후 업데이트하게 변경한다.

### SAC : Soft Actor-Critic
Soft Actor-Critic은 Maximum Entropy RL 원리를 도입해 탐색과 안정성을 동시에 잡은 알고리즘이다.
DPG에서는 $\epsilon-\text{greedy}$ 방식을 exploration 전략으로 사용했던 것에 반해, SAC에서는 entropy를 최대화하는 방향으로 policy를 학습하게 한다. Entropy가 커지면 policy에 무작위성이 더욱 부여되는 것이므로 exploration을 풍부하게 할 수 있다.
우선 SAC에서는 RL의 목적함수가 다음과 같이 바뀐다.
$$J(\theta)= \mathbb{E}_{\tau\sim\pi_\theta(\tau)}\left[\sum_t r(s_t, a_t) + \alpha H(\pi(\cdot \mid s_t))\right]$$
이제 policy는 이 $H(\pi) = \mathbb{E}_{\tau\sim\pi(\tau)}\left[-\log\pi(a_t \mid s_t)\right]$ 를 최대화하는 방향으로 학습하게 된다. 이렇게 해서 Entropy가 커지면 위에서 말했듯이 policy에 무작위성이 더욱 부여되어 다양한 solution에 대해 학습할 수 있어 robust한 policy를 만들 수 있고, 이것이 exploration을 더 할 수 있게 한다. 이에 따라 local optima에 빠지는 것을 막을 수 있고 안정적으로 학습할 수 있게 된다.
또, value function과 관련해서 단순히 Q를 평가하는 대신 엔트로피 보너스($\alpha H(\pi)$) 를 추가한다. 이를 통해 기존에 가치만을 고려하던 것을 넘어서 엔트로피(다양성)까지 고려한다.