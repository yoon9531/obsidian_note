#### Code structure 
`run_hw3.py` 
- 환경 생성
- `DQNAgent` 생성
- 학습 루프 생성
- 평가 및 로그 저장
`agents/dqn_agent.py`
`env_configs/dqn_basic_config.py` :  사용될 Q-network 구조, 하이퍼파라미터 (learning rate, gamma 등), 프레임 스택 등을 설정
`infrastructure/replay_buffer.py` : replay buffer 구현

#### 학습 흐름
`run_hw3.py` $\rightarrow$ `run_training_loop()` $\rightarrow$ `DQN agent` $\rightarrow$ 환경과 상호작용
$\rightarrow$ ReplayBuffer로 경험 저장 $\rightarrow$ Q-network 학습

### Trick
#### Exploration scheduling for ε-greedy action
DQN은 보통 $\epsilon-greedy$ 탐험 전략을 사용하여 액션을 선택한다. 즉, 확률 $\epsilon$으로 무작위 액션을 선택하고 $1-\epsilon$의 확률로 Q-value가 높은 액션을 선택한다.
`exploration_schedule()` : $\epsilon$ 값을 지수적으로 감소시킨다. 
이는 초기에는 Q-network가 정확하지 않기 때문에 많은 탐험을 하는 것이 중요하지만, 학습이 진행될 수록 잘 학습된 정책을 더 많이 이용하는 게 더 낫기 때문이다.
#### Learning rate scheduling
학습 초반에는 큰 learning rate로 빠르게 학습하고, 후반에는 작은 learning rate로 세밀하게 조정하는 방식. `DQNAgent.lr_scheduler` 에 구현돼있음.
#### Gradient clipping
학습 도중 gradient norm이 너무 커지면, 네트워크가 발산할 수 있기 때문에 norm이 일정 threshold를 넘으면 gradient를 잘라낸다. 특히 Q-learning은 target과 prediction 사이의 차이가 클 수 있기 때문에 gradient가 폭발적으로 상승할 가능성이 있따.
#### Atari Wrappers
DQN에서 Atari 환경을 잘 처리하기 위해 아래와 같은 전처리를 한다.
- Grayscale
	- RGB 이미지 -> Grayscale 이미지 : 연산량은 감소하지만 정보 손실은 없음
- Frame-skip
	- 동일한 액션을 4프레임 동안 유지하고, 중간 프레임은 무시
- Frame-stack
	- 마지막 4개의 프레임을 스택(84×84×4)
		➜ 현재 상태뿐 아니라 최근 움직임 정보(속도, 방향)를 함께 인식하게 함
		➜ DQN은 순환 구조(RNN 등)가 없기 때문에 이전 정보 제공이 필요

`update_critic()`