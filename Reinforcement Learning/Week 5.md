지금까지 Behavior cloning에서 시작해서, policy gradient 그리고 이를 최적화하는 reward to go, baseline, actor-critic algorithm을 알아보았고 actor critic alogirthm에서 critic update하는 과정을 최적화 하기 위해 N-sep return, Importance sampling 방식을 알아보았다. 또, 학습하는 policy가 너무 커지는 것을 방지 하기 위해 clipping을 사용해 이를 방지하는 PPO를 알아보았다. 하지만 이 actor-critic algorithm은 $V, Q$ 업데이트하고 policy gradinet로 policy update하고 또 $V,Q$ 업데이트하고 policy 업데이트하는 과정을 거친다. 이를 개선하기 위해 정책을 직접 미분하고 update하여 분산을 큰 문제가 발생하는 policy gradient 방식보다는 정책을 파라미터화 해서 미분하지 않고도, $Q$를 알고 있다는 가정 하에
$\arg\max_{\theta} Q^\pi(s,a)$를 통해 바로 새로운 정책 $\pi$를 계산할 수 있다. 이를 위해 
## Q-learning
- Q-learning은 value-based 강화 학습 알고리즘이다. 목표는 상태 $s$에서 행동 $a$를 취했을 때 얻을 수 있는 누적 보상의 기댓값을 추정하는 함수 $Q(s,a)$를 학습하는 것이다.
- 최적 정책을 정의하는 방법은 다음과 같다.
	$$
	\pi(s) = \text{argmax}_a Q(s,a) 
	$$
- Q-learning은 bellman equation을 근사한다.
	$$
	Q_\phi(s,a) \approx r(s,a) + \max_{a^\prime}Q_\phi(s^\prime, a^\prime)
	$$
- 즉, Fitted Q-learning 알고리즘은 다음과 같다.
	1. 임의의 정책으로 환경에서 데이터를 수집한다.
	2. Bellman update로 target을 계산한다.
		$$ y_i=r_i + \gamma\cdot \max_{a^\prime}Q_\phi(s^\prime, a^\prime)$$
	3. 현재 Q함수에 대해 회귀한다.(supervised learning)
		$$Q_\phi(s_i, a_i) \leftarrow y_i$$
	4. 정책을 개선한다.
		$$ \pi(s) = \text{argmax}_a Q_\phi(s,a)$$
- 이 방식의 장점은
	1. Off-policy : 어떤 정책으로 데이터를 수집하든 사용 가능하다. 
	2. 정책은 $Q$에서 암묵적으로 정의되므로 파라미터를 명시적으로 업데이트하지 않아 policy gradient를 사용하지 않아도 된다.
	3. 구현은 간단하지만 성능이 좋다.
- 하지만, 이 방식은 boostraping으로 인해 target이 움직이고 동작을 위해 많은 trick이 들어가기 때문에 수렴이 보장되지 않는다. 또, 그냥 policy에 비해 학습이 어려울 수 있으며, Q-value를 모든 가능한 continuous 행동공간에서 평가하는 것은 불가능하다.
- 또, Agent가 시간 순서대로 환경과 상호작용하며 transition들을 수집하고 이 수집한 데이터들은 서로 강하게 연관돼있다.(correlated). 신경망은 i.i.d 데이터를 가정하고 학습하기 때문에 이런 상관성은 학습을 불안정하게 만들고 수렴을 방해한다. 
	- 이를 해결하기 위해 Replay buffer를 생성한다.
	- 이전 경험들을 replay buffer에 저장하고 무작위로 샘플링해서 학습에 사용한다. 이렇게 하면 데이터를 decorrelate할 수 있어 학습 안정성이 증가한다.
- Q- learning에서 다음 타겟을 계산할 때 사용하는 $Q$값을 계싼할 때는 같은 네트워크에서 나온 $Q$를 사용한다. 그러면 타겟도 움직이고 파라미터도 동시에 업데이트 되므로 학습이 불안정하거나 발산할 수 있다.
	- 따라서 타겟 계산에 사용되는 $Q_{\theta_{target}}$을 복사본으로 만들어 일정 주기마다 업데이트 한다.
- Q-learning은 현재 Q값이 높은 행동만 계속 선택하는 greedy한 방법이다. 따라서 충분히 exploration하지 못해 local optima에 빠질 수 있다. 
	- Epsilon-Greedy를 택한다. 
	- 확률 $\epsilon$으로 무작위 행동을 선택한다.
## Deep Q-Network (DQN)
DQN은 correlated example, moving target 문제를 해결하기 위해 등장한 개념이다.
- Correlated samples $\rightarrow$ replay buffer
- Moving target $\rightarrow$ Target network
전체 학습 단계는 다음과 같다.
1. Save target network parameters
	- 타겟 네트워크의 파라미터를 복사해서 저장한다.
	- $\phi$ : 현재 학습 중인 Q-network의 파라미터. 계속 업데이트 된다.
	- $\phi^-$ : target network parameter. 타깃을 안정적으로 업데이트하기 위해 일정 스텝마다 복사한다.
	$$ \phi^{-} \leftarrow \phi$$
2. N-step 동안 환경과 상호작용하여 데이터를 수집하고 Replay buffer에 저장한다.
	- N-step만큼 $(s_i, a_i, s_i^\prime, a_i^\prime)$ 수집하고 Buffer $\mathcal{B}$에 저장한다.
3. Replay buffer에서 미니 배치를 무작위로 샘플링한다.
	- K번 반복하며
	- 매번 B에서 무작위 k개의 transition $(s_i, a_i, s_i^\prime, a_i^\prime)$를 추출한다.
4. Loss를 최소화하는 방향으로 Q-network를 업데이트한다.
	- 타겟값 계산
	$$y_i =r(s_i,a_i)+\gamma \max_{a^{\prime}}Q_{\phi^{-}}(s_i^\prime, a^\prime)$$
		- $\max_{a^{\prime}}Q_{\phi^{-}}(s_i^\prime, a^\prime)$ : target network로 예측한 다음 상태의 최대 Q값
	- 손실 함수
	$$L(\phi)=\sum_i ||Q_\phi(s_i,a_i)-y_i||^2$$
		- $Q_\phi(s_i,a_i)$ : 학습 중인 네트워크가 예측한 $Q$값
		- 왜 network 인가? : 함수 근사 도구로 신경망을 쓰기 때문이다.
이렇게하면 inner loop안에서는 target이 변하지 않아 안정적이다.
#### Explore new actions
Q-learning 기반의 알고리즘은 기본적으로 Q값이 높은 행동을 계속 선택하는 greedy한 방법을 사용한다. 하지만 이는 충분한 exploration을 할 수 없다는 단점이 있고 local optima에 빠질 수 있다는 문제가 있다. 따라서 충분히 exploration을 할 수 있도록 설계해줘야 한다.
이 방법으로 Epsilon greedy 방법이 있다.
- $\epsilon-\text{greedy}$
	- 확률 $\epsilon$으로 무작위 행동을 선택하고
	- 확률 $1-\epsilon$으로는 최적의 행동 $a^* = \text{argmax}_a Q(s,a)$를 선택한다.
	- 보통 초기에는 $\epsilon$을 높게 시작하여 넓게 탐험하고 학습이 진행됨에 따라 $\epsilon$값을 점점 줄이게 된다. 이를 epsilon-decay라고 한다.
### Improvements over DQN
이제 이렇게 이 DQN을 더 개선할 수 있는 방안을 살펴볼 것이다.
먼저 DQN의 한계와 개선점을 살펴보면 다음과 같다.
- **Q-value 과대추정(overestimation)**: max 연산 때문에 실제보다 높은 Q값이 학습됨
     $\rightarrow$ Double DQN
- **Q-target이 부정확함**: bootstrapped target이 불안정
	    $\rightarrow$ n-step Q-learning
- **모든 샘플을 균등하게 학습**: 중요한 transition이 덜 학습될 수 있음
	    $\rightarrow$ Prioritized Experience Replay (PER)
- **정확한 Q분포를 알 수 없음**: 단일 기대값만 예측함
		$\rightarrow$ Distributional RL

1. Double DQN
	- 기존 DQN은 max를 target과 network 둘 다에 사용해 Q-value가 과대추정된다. 따라서 Double DQN 방법에서는 행동 선택과 값 추정을 분리해서 수행한다. 즉, 현재 네트워크로 행동(action)을 선택하고 target 네트워크로 그 행동 값(action value)을 추정한다.
		$$ y_i = r_i + \gamma Q_{\phi^{-}}(s^\prime, \text{argmax}_a Q_\phi(s^\prime, a))$$
2. n-step Q-learning
	- 기존 Q-learning은 1-step reward만 반영한다.
	- n-step Q-learning에서는 미래 보상 여러 개를 누적해서 사용한다.
	- 이를 통해 reward-to-go를 더 풍부하게 반영할 수 있고 학습이 빨라져 target variance도 줄어든다.
	$$ y_i = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n \max_a Q(s_{t+n}, a)$$
3. Prioritized Experience Replay(PER)
	- 유용한 transition을 더 자주 학습하는 방법이다. 
	- 기존 replay buffer는 무작위로 transition을 sampling 했는데에 반해
	- PER은 TD-error가 큰 transition일수록 더 중요하다고 판단하고 샘플링 확률을 높인다.
	- 이를 통해 학습을 더 빠르고 효율적으로 진행할 수 있고 중요한 학습 기회를 놓치지 않을 수 있다.
	$$ P(i) \propto (\text{TD-error}_i + \epsilon)^\alpha$$
4. Distributional RL
	- Q-value 분포 자체를 학습하는 방법이다.
	- 기존 Q-learning은 기댓값 $\mathbb{E}[Q]$만 학습했는데
	- Distributional RL은 보상의 분포 $Z(s,a)$를 직접 모델링하여 불확실성을 표현하고 risk-sensitive한 정책을 설계할 수 있다.
	$$ Z(s,a) \sim \text{distribution of returns}$$
	