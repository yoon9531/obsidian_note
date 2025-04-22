Policy wants to try something better than given dataset $\rightarrow$ good bad?

Out of distribution action 
-> keep current policy close to behavior policy(dataset) 
problem : pessimistic ex($\pi_\beta$ is ranodm policy)

Conservative
unseen actino-> under estimate Q-value
### Offline RL
> 이미 수집되어 주어진 $(s_i, a_i, r_i, s_{t+1})$ 만 사용한다. 즉, 새로운 환경 상호작용 없이 주어진 데이터로만 policy, value function을 학습시킨다. $\rightarrow$ static dataset 사용

> [!Offline RL algoorithms]
> - **Imitation Learning**
> 	- Behavior Cloning (BC)
> 	- DAgger
> - **Batch RL**
> 	- CQL(Conservative Q-Learning)
> 	- IQL(Implicit Q-Learning)
> 	- AWR(Advantage-Weighted Regression / Actor-Critic)

- Data가 unknown policy $\pi_\beta$로부터 수집되는데, 이 $\pi_\beta$를 behavior policy라고 한다.
	- $s\sim\pi_\beta(\tau)$
	- $a\sim\pi_\beta(\cdot \mid s)$
	- $s^\prime \sim p(\cdot \mid s, a)$
	- $r=r(s,a)$ 
하지만, Offline RL의 목적도 결국 보상을 최대화 하는 것이고 이를 달성하기 위해 parameter $\theta$를 학습시킨다.
$$
\max_\theta \sum_t\mathbb{E}_{\tau\sim\pi_\theta(\tau)}[r(s_t, a_t)]
$$
이 offline RL은 왜 필요할까? 우선 Online RL은 데이터 수집 비용이 비싸다. 매번 업데이트된 policy를 통해 새로운 trajectory를 탐험하고 또 이를 학습시키는 과정을 반복해야하기 때문이다. 또, 이는 overestimation 문제가 존재하고 다양한 경우에 대해서 학습하는 데에 제한이 있다. 따라서, 항상 있는 데이터를 재사용하는 것은 중요하고 효율적이다. 이전에도 이와 같은 방법을 적용하기 위해 replay buffer를 도입한 적이 있다. 하지만, offline RL은 주어진 dataset에 없는 action을 잘 다뤄야 하는 어려움이 있다. 
### Online RL
> Agent가 현재 정책을 써서 환경과 상호 작용하여 수집한 데이터로 policy, value function을 학습시킨다. 즉, 매 time step 혹은 episode마다 최신으로 수집한 데이터(On-policy or Off-policy)로 정책, 가치 함수를 업데이트하는 방식이다.

>[!Online RL algorithms]
>- **On-policy**
>	- Actor-Critic (A2C / A3C)
>	- TRPO
>	- PPO
>	- Naive policy gradient
>- **Off-policy**
>	- Q-learning / SARSA
>	- DQN $\rightarrow$ DDQN / Prioritized ER / Rainbow
>	- DDPG (Deep Deterministic Policy Gradient)
>	- TD3
>	- SAC

#### On-policy vs Off-policy
- On-policy : 데이터 수집 정책과 학습하는 정책이 같다. (PPO, A2C)
- Off-policy : 과거 Buffer, 다른 정책 data도 재활용 하여 샘플 효율 극대화 (DQN, SAC ...)

#### Offline RL vs BC
![[Pasted image 20250418145858.png | 600]]
- *Behavior Cloning*은 
	- Expert Data로 SL처럼 학습하기 때문에 expert data에 전이가 A에서 B로 가라 혹은 B에서 C로 가라 이렇게 나와있으면 너도 A에서 B로 가라, 그리고 B에서 C로 가라 하는 것이다. 즉 각 전이에 대해 그대로 따라하는 것이기 때문에 ==A에서 C로 바로 가는 행동에 대해서는 배울 수 없다.==
	- 데이터에 나쁜 행동, 비효율적인 행동이 섞여있어도 **그대로 믿고 배운다.**
- *Offline RL*은
	- A$\rightarrow$B, B $\rightarrow$ C를 보상 정보와 함께 학습한 후 'A에서 C로 가는 것이 더 보상을 높을 것 같다'라고 하는 계산을 할 수 있기 때문에 A $\rightarrow$ C 경로를 발견할 수 있다.
	- 즉, A에서 취할 수 있는 **모든 행동** 중 가장 높은 Q-value를 찾아 **데이터에 없던 경로까지 발견할 수 있다.(stitch)**
	- 보상에 따라 나쁜 행동에는 낮은 Q-value, 좋은 행동에는 높은 Q를 붙인다.
	- 또, 효율이 낮은 행동에 대해서도 가치 함수 학습에 사용하기 때문에 다양한 상황을 평가하고 최적의 정책을 찾아낼 수 있다.
	- 그렇지만 Offline RL은 어렵다. 다양한 상황을 고려하기 때문에 **나쁜 행동에 대해서도 시도할 수 있고 이것이 높은 보상을 가져온다면 이런 행동을 선택할 수 도 있다**. 만약, 차가 핸들을 꺾어야 하는지 벽을 박고 가야 하는 지에 대한 상황을 가정했을 때, 목적지로 가는 경로는 벽을 박고 가는 것이 더 빠르기 때문에 해당 행동에 대한 보상을 높게 줘 이 행동을 선택할 수 있다. 하지만 이는 나쁜 행동이므로 학습에 사용하기 적절한 데이터라고 할 수 없다. 따라서 우리는 이 정책이 학습할 우리가 모르는 행동에 대해 안전하게 처리할 수 있는 방안이 필요하다. 
#### Overestimation of Q-value in Offfline RL
Q-learning에서 target 값은 다음과 같이 업데이트 된다.
$$
y_i \leftarrow r_i + \max_{a_i^\prime} Q_\phi(s_i^\prime, a_i^\prime)
$$
이때 $\max_{a_i^\prime} Q_\phi(s_i^\prime, a_i^\prime)$에서 관측하지 않은 행동에 대한 $Q$값 까지 계산하게 된다. 우리는 Offline RL을 수행하기 때문에 관찰하지 않은 $a^\prime$ 때문에 Q값이 과장되어 overestimate될 수 있다. 왜냐하면 관측하지 않은 행동에 대해 아무런 보정 없이(penalize 등) $\max$값을 구하면 자기 맘대로 높은 값을 예측할 수 있고 그런 허상을 골라버리게 된다. 이를 다음 그래프로 살펴보자
![[Pasted image 20250418151438.png|400]]
여기서 data support, 즉, 데이터가 주어진 상황에서는 Q의 예측 값이 잘 들어맞는다. 하지만 OOD(out of distribution) 행동(관측되지 않은)들에 대해서는 사실상 검증이 불가하다. 따라서 아무 높은 값이나 예측하면 되는 것이다. 따라서 overestimate된 Q를 선택하게 되는 것이다.
#### Mitigate overestimation
##### 1. Constriain $a^\prime$ to stay close to behavior policy
**학습하는 policy $\pi$가 행동 데이터 분포 $\pi_\beta$에서 크게 벗어나지 않도록 제한**하는 방식이다. 하지만 우리는 $\pi_\beta$에 대한 정보가 없다. 그냥 모른다(random policy). 그런데, 주어진 데이터는 $\pi_\beta$에 의해서 만들어진 것이므로 $\pi$를 이 데이터에 맞추면 된다. 따라서 데이터를 BC로 근사시키면 된다.
$$
\pi = \arg\max_\pi\mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\lambda Q(s, \pi(s)) - (\pi(s) - a)^2\right]

$$
여기서 $(\pi(s) - a)^2$은 BC에 근사시키는 것과 관련있는 식이다. $a$가 주어진 데이터의 행동이고 $\pi(s)$가 상태 s가 주어졌을 때 우리의 policy가 선택할 action일 것이다. 만약 이 $\pi$를 통해 선택한 행동이 주어진 데이터의 행동과 크게 다르다면 Q값은 낮아져 선택되지 않을 것이고, 가깝다면 Q값이 그만큼 낮아지지 않아 선택될 확률이 높아진다. 하지만 이 방식만으로는 한계가 있다.
1. _너무 비관적이다._
	- 만약 $\pi_\beta$가 random policy이면 사실 상 아무것도 배우지 못할 수도 있다.
2. _충분히 비관적이지 않다._
	- 또 $\pi_\beta$가 충분히 괜찮은 policy라고 해도, $\pi$가 여기서 조금만 벗어나도 OOD 행동을 취할 수 있고 이로 인해 Q값을 overestimate할 수 있다.
--- 
이를 해결하기 위한 방안으로는 다음 두 가지를 생각해볼 수 있다.
1. Q를 보수적으로 학습한다.
		![[Pasted image 20250418154742.png | 600]]
	- 이 방식은 **Conservative Q-Learning (CQL)** 방식이다.
	- **OOD $a^\prime$에 penalty를 주고 supported data 내 행동에 대해서는 보상을 최대화**하는 정규화 항을 더해 Q를 학습한다.
	- 이를 통해 Q값을 OOD 영역에서는 낮게 학습하게 되고 $\pi$가 조금 벗어나도 해당 행동의 $Q$가 낮아 학습하지 않게 된다.
2. 아예 _OOD에 있는 행동 $a^\prime$에 대해서는 학습하지 않는다._
	- Supported data에 속하는 데이터만 학습에 포함시키는 방식이다.
	- **IQL(Imitation Q-Learning), BCQ**

#### Conservative Q-Learning
다음 식을 통해 Q를 추정할 수 있다.
$$
\hat Q^\pi = \arg\min_Q \max_\mu \mathbb{E}_{(s,a,s^\prime)\sim D} \left[\left(Q(s,a) - (r(s,a) + \gamma\mathbb{E}_\pi[Q(s^\prime, a^\prime)]) \right) \right] + \alpha\mathbb{E}_{s\sim D, \mathbf{a} \sim\mu(\cdot \mid s)}[Q(s,a)]-\alpha\mathbb{E}_{(s,a)\sim D}[Q(s,a)] + R(\mu)
$$
이 식을 3가지 부분으로 나눠서 볼 수 있다.
1. TD 손실
	- 여기서 $\mathbb{E}_{(s,a,s^\prime)\sim D} \left[\left(Q(s,a) - (r(s,a) + \gamma\mathbb{E}_\pi[Q(s^\prime, a^\prime)]) \right) \right]$는 Critic 손실이고 이는 Q값과 Bellman target 차이를 제곱해 최소화 한다.
2.  OOD 억제
	- 또, $\alpha\mathbb{E}_{s\sim D, \mathbf{a} \sim\mu(\cdot \mid s)}[Q(s,a)]$는 conservation(보수성) penalty이다. 이는 supported data 밖의 행동 $\mu$에 대해 Q를 낮게 만들도록 밀어내리도록 하며 실제 데이터 행동에 대해서 Q값을 지나치게 낮추지 않도록 보상해준다. 이는 OOD 행동은 Q를 내리고 데이터 행동에 대해서는 Q를 그대로 두게 하기 위한 것이다.
3. 데이터 회복 
		$$-\alpha\mathbb{E}_{(s,a)\sim D}[Q(s,a)]$$
	- 실제 관측된 $(s,a)$에 대해서는 동일한 가중치로 Q값을 낮추지 않도록 보상해준다.
4. 엔트로피 정규화
	- $R(\mu) = \mathbb{E}_{s\sim D} [\mathcal{H}(\mu(\cdot \mid s))]$ 를 더해서 $\mu$에 대한 closed form solution을 얻을 수 있다. 
	- 이를 추가해주지 않았을 때, 이전에 가장 Q가 크게 나올 것 같은 행동 분포 $\mu$를 골라서 그 분포 하에 Q값을 높이는 벌점을 주겟다는 뜻이었다. 하지만 이때 **$\max_\mu$ 최적화는 무한히 많은 행동 분포를 탐색해야 하기 때문에 사실상 불가능에 가깝다고 할 수 있다**. 하지만 이를 추가함으로써 엔트로피를 높이는 방향으로 페널티를 완화시킬 수 있따.
이제 원래 OOD를 억제하는 항과 합쳐지면 다음과 같이 LogSumExp 형태로 바뀌면서 데이터 지원 밖 모든 행동에 대해 Q값을 일괄적으로 억제해주는 구현이 된다.
$$
\begin{aligned}
&\;\alpha\,\mathbb{E}_{s\sim\mathcal D,\;a\sim\mu^*(\cdot\mid s)}\bigl[Q(s,a)\bigr]
\;-\;\alpha\,\mathbb{E}_{(s,a)\sim\mathcal D}\bigl[Q(s,a)\bigr]
\;+\;\alpha\,R\bigl(\mu^*\bigr)\\
&\quad=\;\alpha\,\mathbb{E}_{s\sim\mathcal D}\Bigl[\log\!\sum_{a} \exp\!\bigl(Q(s,a)/\alpha\bigr)\Bigr]
\;-\;\alpha\,\mathbb{E}_{(s,a)\sim\mathcal D}\bigl[Q(s,a)\bigr].
\end{aligned}
$$
따라서, 전체 손실함수 $L_{CQL}$을 다음과 같이 쓸 수 있게 된다.
$$
\begin{aligned}
L_{\mathrm{CQL}}(Q)
&=\;\underbrace{\mathbb{E}_{(s,a,s')\sim\mathcal D}
  \Bigl[\bigl(Q(s,a)\;-\;(r(s,a)+\gamma\,\mathbb{E}_{a'\sim\pi}[Q(s',a')])\bigr)^2\Bigr]}_{\text{표준 TD 손실}}\\
&\quad+\;\alpha\,\underbrace{\mathbb{E}_{s\sim\mathcal D}\Bigl[\log\!\sum_{a}\exp\!\bigl(Q(s,a)/\alpha\bigr)\Bigr]}_{\substack{\text{OOD 행동 억제}\\\text{(LogSumExp 항)}}}
\;-\;\alpha\,\underbrace{\mathbb{E}_{(s,a)\sim\mathcal D}\bigl[Q(s,a)\bigr]}_{\text{데이터 행동 회복}}.
\end{aligned}
$$
하지만 위 두 방식에서도 $\pi(s)$나 $a^\prime$, $a\sim\mu(\cdot \mid s)$를 볼 수 있다. 이를 통해 우린 아직도 OOD action에서 평가된 값(inaccurate)을 사용한다. 또, 간단한 아이디어고 실전에서 잘 작동하지만 $\alpha$를 tuning하는 것이 중요하고 log-sum-exp를 측정하고 BC pre-training과 같은 요구 사항이 존재한다.
그럼 이제 우리는 다음과 같은 질문을 던질 수 있다.
`OOD action의 값을 아예 측정하지 않는 방법은 없을까?`
이에 대한 답을 이제 알아보자

#### Behavior Cloning with Rewards
Behavior cloning의 문제점은 **좋은 trajectory나 나쁜 trajectory나 똑같이 취급**하는 것이다. 즉, Behavior cloning은 expert data를 가지고 학습을 진행하는데 해당 데이터에 bad trajectory가 들어있어도 이를 그냥 학습한다는 것이다. 
이를 해결하는 방법은 좋은 trajectory만 선별해서 모방하는 것이다 이를 **Filtered Behavior cloning**이라고 한다. trajectory들의 reward를 계산해 순위를 매긴다($r(\tau) = \sum_t r(s_t, a_t)$ ). 이 중에서 가장 높은 K%의 trajectory만 선별하고 이를 dataset ($\mathcal{D}$)으로 사용하여 policy를 학습시킨다.
$$
\arg\max_\pi \mathbb{E}_{(s,a)\sim D}[\log\pi(a\mid s)]
$$
여기서 $D$는 선별된 dataset이다.
하지만 이거에도 문제가 있다. 
1. 모든 **transition을 동일하게 취급**한다는 것이다. 
	- 이전에는 trajectory가 좋은지 나쁜지를 구분했다면 이제 transition이 좋은지 나쁜지를 구분할 차례이다.
2. 또, 보상을 좋은 지 나쁜 지로만 판단한다.
	- 얼마나 좋은 지를 생각하지 않고 이 기준 넘으니까 좋은거 이 기준보다 낮으니까 나쁜거 이렇게 해버린다는 것이다(이분법적).
	- 이보다 상대적으로 이 행동이 얼마나 더 좋은지를 생각하는 것이 더 좋다(상대적).
이를 해결하기 위해 각각의 **transition에 대해 advantage를 계산**하여 얼마나 이 action이 더 좋은 지를 알 수 있게 weight를 추가한다.
이를 Advantage-Weighted Regression(AWR)이라 한다.
#### Advantage-Weighted Regression
이는 Behavior cloning을 advantage와 함께 사용하는 것과 같다.
1. Fit $\hat V^{\pi_\beta}$ 
	$$\arg\min_V \mathbb{E}_{(s_t, a_t)\sim D}[(R_t(\tau)-V(s_t))^2$$
	- 여기서 $R_t(\tau)$는 trajectory $\tau$에서 t이후 얻은 모든 return이고(Monte carlo return)
	- $V(s_t)$ 를 이 상태에서 앞으로 기대되는 리턴으로 예측하도록 학습시킨다.
	- 이를 통해 Advantage $A(s_t, a_t)= R_t(\tau) - V(s_t)$를 계산할 수 있다.
2. Train $\hat \pi$
	$$\arg\max_\pi\mathbb{E}_{(s_t, a_t)\sim D}\left[\log\pi(a_t \mid s_t\exp(\beta(R_t(\tau)-\hat V^{\pi_\beta}(s_t)))\right]$$
	행동 $a_t$를 모방할 확률 $\pi(a_t\mid s_t)$에 $\exp(\beta A(s_t, a_t))$만큼의 가중치를 준다. $\beta$는 차이가 얼마나 큰 가중치를 강조할 지를 나타내는 hyperparameter이다. 이는 advantage가 양수이고 클 수록 행동을 더 자주 학습하게 되고 음수가 되면 해당 행동은 학습하지 않게 된다. 이렇게 하면 OOD action에 대해서는 가중치가 작아져 실질적으로 학습에 적용되지 않는다. 또, policy gradient를 사용하지 않고 두 번의 regression 만으로 학습이 가능하다는 단순성이 있다.
	여기서도 문제점이 있다.
	1. $R_t(\tau)$의 분산이 크므로 가중치가 불안정할 수 있다.
	2. $V^{\pi_\beta}$를 학습할 때 behavior policy를 사용하기 때문에 $\pi_\theta$의 advantage보다는 약할 수 있다.
#### SARSA style objective
SARSA style을 사용해서 OOD action을 사용하지 않고 Q-learning을 하는 방법이다. 기존 Q-learning의 loss function은 
$$L_Q(\phi) = \mathbb{E}_{(s,a, s^\prime) \sim D}\left[(Q_\phi(s,a)-(r(s,a) + \gamma\mathbb{E}_{a^\prime\sim\pi(\cdot\mid s^\prime)} [Q_{\phi^\prime}(s^\prime,a^\prime)])\right]$$ 이다. 이를 최소화 하는 것이 Q learning의 목표였다. 하지만 이는 여전히 $\pi$를 사용한다. 즉 OOD action을 사용한다는 것이다. 따라서 다음과 같은 SARSA style로 바꾸면 loss function은
$$L_Q(\phi) = \mathbb{E}_{(s,a,r,s^\prime, a^\prime)\sim D}\left[(Q_\phi(s,a)-(r+\gamma Q_{\phi^\prime}(s^\prime, a^\prime)))^2\right]$$ 이다. 이렇게 해서 Q 값을 OOD action을 사용하지 않고 학습시킬 수 있다.
하지만 여기서도 $Q_{\phi^\prime}$은 dataset의 좋은 행동과 나쁜 행동을 구분하지 않고 모두 사용한다. 하지만 $\pi$ 는 $\pi_\beta$로 부터 좋은 action만 학습해야 하기 때문에 다음과 같이 식을 변형시켜 좋은 행동만 학습할 수 있도록 한다.
$$L_Q(\phi) = \mathbb{E}_{(s,a,r,s^\prime, a^\prime)\sim D}\left[(Q_\phi(s,a)-(r+\gamma \max_{a^\prime \in A \ s.t. \pi_\beta(a^\prime\mid s^\prime)>0} Q_\phi(s^\prime, a^\prime)))^2\right]$$

#### Implicit Q-Learning (IQL)
