## Reinforcement Learning
> [!Wikipedia]
> Reinforcement learning (RL) is … how an intelligent agent ought to take actions in a dynamic environment in order to maximize the cumulative reward

그렇다면 이를 우리는 어떻게 수학적으로 표현할 수 있을까?(formalize)
강화 학습은 누적 보상을 최대화 하기 위해 agnet가 변화되는 환경에서 어떤 행동을 취해야 하는지 학습하는 것이다. 이를 수학적으로 나타내면 
$$
Find \ \pi(a_t \mid s_t) \ \ maximize \ \ \sum r_t
$$
이다. 즉, 강화 학습을 진행하는 과정은
1. 현재 상태 $s_t$에서
2. policy $\pi_\theta(a_t \mid s_t)$를 통해 어떤 행동 $a_t$를 결정하고
3. 행동의 결과로 다음 상태 $s_{t+1}$를 $p(s_{t+1} \mid s_t,a_t)$를 통해 확률적으로 정하고 해당 상태에 대한 보상 $r_t$를 준다.
4. 이게 매 step 마다 반복돼서 다음 trajectory를 생성한다.
$$
(s_0, a_0, r_0), (s_1, a_1, r_1), \dots
$$
이 모든 것을 묶은, 즉, 강화 학습 환경을 정의하는 수학적인 틀을 **Markov Decision Process(MDP)** 라고 한다. 여기서 environment가 의미하는 바는 이전에 말한 environment와는 다소 다르다. 여기서 environment는 
$$ environment = environment + task(reward)$$ 을 의미한다.

먼저 Markov Decision Process에서 Process의 의미를 보면
#### Process
process는 시간에 따라 변화하는 무작위 변수들의 모음으로 수학적으로는 다음과 같이 표현할 수 있다.
$$ \{s_t\}_{t\in T=[0, \infty]}$$
이 식의 의미는 시간 $t$에 따라 상태 $s_t$가 정해지는데, 그 상태가 확률적으로 정해진다는 의미이다. 즉, 이를 강화 학습의 관점에서 살펴보면 환경은 에이전트의 행동, 환경의 반응에 따라 달라지는데 이렇게 달라질 때마다 에이전트가 환경에 따라 관찰하는 상황 $s_t$가 달라지는 것이다. 이제 Markov process를 이해할 차례이다. 
#### Markov Process
이는 process는 process인데 Markov Property를 만족하는 process이다. 이는 미래 상태 $s_{t+1}$이 현재 상태 $s_t$에만 의존하고 과거 상태에는 의존하지 않는 property이다. 따라서 만약 $s_t$가 Markov process라면 다음 필요 충분조건을 만족한다. $$p(s_{t+1} \mid s_t) = p(s_{t+1}\mid s_1, \dots, s_t)$$ 그리고 만약 상태 공간과 시간이 _이산적이라면 이를 Marko chain_ 이라고 부르기도 하며 수학적으로는 다음 tuple로 나타낼 수 있다. $$\left<S, p(s_{t+1} \mid s_t)\right>$$
이 Markov process에 보상의 개념을 추가한 것을 **Markov Reward process**라고 부르고 정책의 개념을 추가한 것이 **Markov Decision Process**라고 부른다.
#### Markov Reward Process (MRP)
이는 Markov Process에 보상 개념이 추가된 것이고 이는 수학적으로 다음 tuple과 같이 나타낼 수 있다.
$$
\left <S, p(s_{t+1}\mid s_t), r(s_t), \gamma) \right >
$$
여기서 $\gamma$는 할인율(discount factor)라고 부르며 0에서 1사이의 값을 가진다. 이는 미래 보상을 현재의 가치로 환산하기 위해 사용된다.
#### Markov Decision Process (MDP)
이는 Markov Process에 정책(policy) 개념이 추가된 것으로 특정 상태에서 어떤 행동을 선택할지를 결정한다. (Recall: 강화학습의 목적 $\rightarrow$ 누적 보상을 최대화하는 것) 이는 수학적으로 다음 tuple과 같이 나타낼 수 있다.
$$
\left<S, A,p(s_0) ,p(s_{t+1}\mid s_t, a_t), r(s_t, a_t), \gamma \right>
$$
이 MDP는 환경을 완전히 정의하며 action을 취하고 전이 확률, 보상을 얻고 다음 state로 넘어가는 일련의과정을 markov decision process라고 한다.
이를 Gymnasium의 Ant 환경으로 예를 들면 $S= \mathbb{R}^{27} \quad$ (torso velocity, height, joint angles, joint velocity), $A=[-1,0, 1, 0]^8$ (joint torque), $p(s_0) = fixed \ pose + uniform \ noise$, $p(s_{t+1} \mid s_t, a_t) = physics \ simulation$, $r(s_t, a_t) = healthy \ reward + forward \ reward - control \ cost$ , $\gamma = 0.99$ 로 들 수 있다. 

 