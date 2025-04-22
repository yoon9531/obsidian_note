### Reinforce learning?
1. Sequential decision making problem
2. Approach for learning decision making
#### Basic Terminologies
1. **Environment** $P(s_{t+1} \mid a_t)$ 
	- 환경은 계속 변하고 있으며 현재 상태 $s_t$에서 에이전트가 행동 $a_t$를 했을 때 다음 상태 $s_{t+1}$이 확률적으로 결정된다. 이를 transition probability(전이 확률)이라고 부르며 $P(s_{t+1} \mid a_t)$로 표기한다
2. **State $s$** (observation $o$, configuration $x$)
	- Agent는 센서를 통해 환경을 관찰하고 이를 통해 얻은 정보를 observation $o$라고 하고, 로봇의 내부 상태와 같은 다른 정보를 포함하면 configuration $x$도 포함된다. 이렇게 얻은 정보로 현재 상태 $s_t$를 결정한다.
3. **Policy $\pi(a_t \mid s_t)$** (agent, strategy, controller)
	- Agent는 상태 $s_t$를 바탕으로 어떤 행동 $a_t$를 할 지 결정해야 한다. 이를 결정하기 위해 사용되는 규칙 혹은 전략을 policy이다. 즉 $\pi(a_t \mid s_t)$는 현재 상태 $s_t$에서 어떤 확률로 $a_t$를 선택할 지를 나타내는 함수다.
4. **Action** $a_t$ (control $u$): 
5. **Trajectory** $(s_0, a_0, s_1, a_1, \dots , s_T)$  (policy roll-out, episode)
	- Agent가 환경과 상호작용을 하며 경험하게 되는 상태 $s_t$와 행동의 $a_t$을 묶어서 나타낸 것이다. 즉, 전체 시간에 대해서 특정 상태에서 policy를 통해 어떤 행동을 결정했고, 이 행동을 통해 다시 어떤 상태로 전이됐는지를 모두 기록해놓은 것이다. 이는 rollout 또는 episode라고도 부른다. 

## Imitation Learning
### Behavior Cloning(BC)
- 지도학습(SL)을 통해 policy를 학습시키는 방법이다. 즉, 사람 또는 전문가가 한 행동들을 모방하여 학습을 하는 방법이다.
- $D = \{(s_0, a_0), (s_1, a_1), \dots \}$ : expert data, demonstration 
학습은 전문가가 보여준 행동을 supervised learning을 통해 학습한다. 목표는 policy $\pi(a \mid s)$를 잘 학습시키는 것이다. 이를 달성하기 위해서는 전문가 데이터와 최대한 비슷한 확률을 출력할 수 있도록 학습하는 것이며 이를 수식으로 나타내면 다음과 같다.
$$
\min_\theta - \mathbb{E}_{(s,a)\sim D}[\log \pi_\theta(a \mid s)]
$$
이는 cross-entropy loss이며 classification에 사용하는 loss이다. 즉, 학습은 이 loss를 최소화하는 방향으로 이루어진다. 즉, 위 수식이 의미하는 바는 expert dataset 에서 무작위로 상태-행동 쌍을 샘플링하여 우리의 policy를 학습시킨다. 상태 $s$에서 parameter를 조정하여 전문가가 하는 행동 $a$에 대해 높은 확률이 나오게끔 해야한다.
Behavior Cloning을 진행하기 위해선 expert data, demonstration이 있어야 한다. 우리는 모방을 통해 학습을 하기 때문에 이 전략에서는 expert data가 존재해야 학습을 진행할 수 있다. 그렇다면 어떻게 이 demonstrations를 모으는 지 알아보자.

#### 1. How to collect demosntrations
- 이미 수집된 데이터 활용 : 사람들이 이미 행동하는 과정에서 기록해 놓은 로그들을 사용할 수 있다.
- Robot의 경우
	- 사람이 직접 관절이나 팔을 물리적으로 움직이며 시연을 제공
	- Remote controller : 원격 조종하며 시연
	- Puppeteering: 로봇을 마치 인형처럼 조작하는 방식
- 그냥 Video로 보면 안되나? $\rightarrow$  안됨
	- Embodiment gap 
		- 외형 차이 : 사람은 다리 둘, 팔 둘 있는데 로봇이 바퀴, 팔 하나 이렇게 있으면 시연이 제대로 이루어지지 않을 수 있다.
		- 물리적 능력 차이 : 사람은 관절 30개 넘게를 자유롭게 사용할 수 있지만 로봇은 6개 가량 밖에 되지 않을 수 있음
	- 하지만 직접적으로 사람이나 동물 데이터를 모방하기는 어려워도 가이드 제공으로서의 역할은 할 수 있다.
#### 2. What can goes wrong
Imitation learning에서 발생할 수 있는 문제점이다.
- **Compounding errors** 
	- 용어에서 알 수 있듯이 이것은 오차가 누적되어 우리가 목표하던 방향의 학습과는 점점 멀어지는 형태로 학습이 진행되는 오류이다. 이것이 어떻게 발생하는 지 알아보자
	- 원래 지도 학습은 예측이 어떻게 나오든 다음 입력 값은 변하지 않는다. 하지만 지금 우리가 진행하려는 Behavior cloning 즉 supervised learning of behavior에서는 예측 값이 다음 입력 값에 영향을 준다.
	- 현재 상태 $s_t$에서 행동 $a_t$를 하면 다음 상태 $s_{t+1}$에 직접적으로 영향을 준다는 것은 직관적으로 이해할 수 있다.
	- 만약 policy가 처음 상태 $s_t$에서 행동 $a_t$를 잘못 예측했을 때, 우리한테 없는 상태인 $s_{t+1}$에 도달할 수 있다. 이 때 policy는 어떻게 해야될 지 모르기 때문에 다시 policy가 다음 행동 $a_{t+1}$를 잘못 에측하게 되고 이렇게 오차가 계속 누적되어 policy $\pi_\theta$가 잘못 학습된다.
	- 이에 따라 전문가가 방문한 상태 분포와 정책 $\pi$가 방문한 상태의 분포와 달라지며 이를 covariate shift라고 부르기도 한다. 
	$$
	p_{expert}(s) \neq p_\pi(s)
	$$
	- 그럼 이를 어떻게 해결할 수 있을까?
		1. 엄청 많이 demo data(전문가의 시연데이터)를 수집한 후에 맞기를 기도하는 기도 메타를 사용하는 방법이다
		2. 학습한 agent가 틀린 행동을 했을 때 어떻게 고쳐야 하는지를 시연해주는 데이터를 추가로 수집하는 방법이다.
- **Multimodal demonstration data**
	- 이는 여러 사람이 demo data를 수집했을 때 발생할 수 있는 문제이다.
	- 만약 경로 찾기 문제에서 한 사람은 왼쪽으로 돌아가는 경로, 한 사람은 오르쪽으로 돌아가는 경로를 선택했다고 하자. 이는 똑같은 목표를 잘 달성했으므로 올바른 demonstration이라고 할 수 있다. 다만, 이를 $L2 \ Loss$로 학습했을 때 문제가 발생한다. $L2 \ loss$는 평균적인 행동을 유도하는 loss이기 때문에 agent는 두 경로 사이를 가려고 하게된다. 하지만 두 경로 사이에 있는 장애물에 부딪히게 되면 목표 달성은 실패하게 된다. 따라서 이 $L2 \ Loss$를 사용하게 되면 multimodal action을 제대로 처리하지 못하는 문제가 발생한다.
	- 이를 해결하기 위해서는
		- 더 표현력 높은(expressive) 확률 분포를 모델링하는 방법을 택할 수 있다.
		- 예를 들어, Gaussian mixture model (GMM), Categorical distribution, VAE (Variational AutoEncoder), Diffusion models를 사용할 수 있다.
- **Mismatch in Observability btw expert & agent**
	- Expert가 agent보다 더 많은 observation을 갖기 때문에 agent가 expert를 정확히 모방할 수 없다. 이는 우리 경험에서 예시를 잘 생각해볼 수 있다. 만약 카톡 내용을 지피티한테 물어본다고 할 때, 지피티는 사람 관계에 대한 전후 사정을 잘 모르지만 expert인 우리는 그 사정과 고려해야할 사항 등에 대해 더 잘안다. 이런 것들이 문제가 되어 agent가 expert를 제대로 모방 및 학습하지 못하는 문제가 발생하는 것이다.
	- 이는 어떻게 해결할 수 있을까?
		- 가장 쉬운 방법으로는 우리가 아는 만큼의 contextual information을 agent한테 주어 정확하게 모방할 수 있도록 하는 것이다.
		- 이는 현실적으로 어려우므로, 다른 방법으로 전문가에게 제한을 두는 것이다. 즉, 전문가도 에이전트와 같이 제한된 정보만을 보게하는 것이다. 이렇게 하면 두 주체 간의 정보 오차로 인한 문제를 해결할 수 있다.
### DAgger
이 알고리즘은 compounding error를 해결할 수 있는 알고리즘이다.
#### 1. DAgger - Expert query
- 이 방식은 학습된 정책이 roll-out 중에 잘못된 행동이 있으면 전문가에게 online으로 잘못된 state에 있을 때 어떤 행동을 선택해야 하는 지를 묻는 방식이다. 이 알고리즘 순서를 요약하면 다음과 같다.
		1. 학습된 정책으로 rollout $\rightarrow$ $\pi_\theta : s_1^\prime, \hat a_1, \dots , s_T^\prime$
		2. 각 상태 $s^\prime$에 대해 전문가에게 이상적인 행동을 질의 : $a^* \sim \pi_{expert}(\cdot \mid s^\prime)$ 
		3. 기존 Dataset에 새롭게 수정된 데이터를 추가한다 : $D \leftarrow D \cup {(s^\prime, a^*)}$
		4. Policy를 update한다 : $\min_\theta L(\pi_\theta, D)$
- 이 방식이 반복 실행되며 점점 policy가 더 나아진다.
- 하지만 이 방식 또한 단점이 존재하는데. 움직이고 있는 agent가 실시간으로 전문가에게 물어보는 것은 어려움이 있다는 단점이 존재한다.
#### 2. DAgger - Expert Intervention
- 이 방식은 이전 DAgger 방식과는 다르게 전문가가 직접 개입하여 full control을 가져와 corrective behavior data를 수집하는 방식이다. 이 알고리즘 순서는 다음과 같다.
	1. 이전 방식 처럼 학습된 정책으로 rollout을 시작한다. $\rightarrow \pi_\theta : s_1^\prime, \hat a_1, \dots , s_T^\prime$
	2. Policy가 실수하거나 잘못된 행동을 하려고할 때 expert가 개입한다.(time : $t$)
	3. 이 이후부터 전문가가 직접 행동을 수행한다. (partial demonstration) : $s_t^\prime, a_t^\ast,\dots, s_T^\prime$
	4. 이 새롭게 시연된 부분을 다시 기존 dataset에 추가한다: $D \leftarrow D \cup {(s_i^\prime, a_i^*); i \geq t}$
	5. Policy를 update한다. : $\min_\theta L(\pi_\theta, D)$
- 이 방식은 human gated DAgger 혹은 traded autonomy, shared autonomy라고도 불린다.
- 이 방식은 실시간으로 일일이 행동을 알려줄 필요 없이 잘못될 것 같으면 그냥 expert가 수행하면 되므로 덜 번거롭다.
- 하지만, 전문가가 빠르게 실수를 감지하고 행동을 해야 하지만 반응이 늦거나 놓칠 수 있는 위험이 존재하며 실전에서는 실수를 감지하기에 너무 빠르게 복잡한 상황이 있을 수 있기 때문에 이러한 환경에서는 제한적으로 동작할 수 있다.

