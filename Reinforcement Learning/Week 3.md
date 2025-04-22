## Policy gradient(정책 경사법)
강화 학습의 목표는 보상의 합을 최대로 하는 policy $\pi(a_t \mid s_t)$ 를 찾는 것이다. 이 policy를 최적화 하는 방법 중 하나가 policy gradient이다. 이는 정책 함수를 parameterize하고 목표 함수의 경사를 따라 파라미터를 업데이트 한다. 즉, policy gradient의 핵심 아이디어는 **높은 보상을 얻는 행동은 더 자주, 낮은 보상을 얻는 행동은 덜 선택되도록** 정책을 조정하는 것이다.
우선 기대 보상을
$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}\sum r(s_t, a_t)$$
로 정의할 수 있고, 우리는 이 $J(\theta)$를 최대화 하는 것이 목표이다.
이를 통해 RL의 목표를 수식으로 나타내면 다음과 같다.
$$
\theta^* = \arg\max_{\theta} 
\underbrace{
\mathbb{E}_{\tau \sim \pi_\theta(\tau)} 
\left[ \sum_{t} r(s_t, a_t) \right]
}_{J(\theta)}
$$
즉, 이를 통해 **보상 합의 기댓값을 최대화하는 parameter를 찾는 것**이 우리의 목표라고 할 수 있다.
이 parameter $\theta$를 구하는 과정을 수학적으로 알아보자.
먼저 우리는 누적 보상합의 기댓값을 $J(\theta)$로 정의했으며, policy $\pi_\theta(\tau) :=p(\tau \mid \pi_\theta)$ 로 정의할 수 있으며 $\tau$는 trajectory를 나타내며 $\sum (s_t, a_t)$ 로 정의할 수 있다. 
우리는 $J(\theta)$를 최대화하는 $\theta$를 찾고 싶지만, 위에서 주어진 식의 상태로는 $J(\theta)$를 $\theta$에 대해 미분한 값을 알 수 없다. $\theta$로 미분해야 하는 이유는 $\theta$에 변화를 주면서 $J(\theta)$를 최대화 시켜야 하므로 $\theta$ 를 어떤 방향으로(?) 변화시켜야 $J(\theta)$가 커지는 지 알기 위해서다. 따라서 $J(\theta)$에 변화를 줘보자
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} r(\tau) = \int \pi_\theta(\tau)r(\tau)\,d\tau
$$
이제 Log-derivative trick을 사용해서 $\theta$로 미분한 값을 알 수 있게 됐다. 여기서 Log-derivative trick은 다음과 같다.
$$
\nabla log\,p(x) = \frac{\nabla p(x)}{p(x)}
$$
따라서 $J(\theta)$의 $\theta$에 대한 gradient는 다음과 같이 구할 수 있다.
$$
\nabla_\theta \, J(\theta) = \int \nabla_\theta\, \pi_\theta(\tau)r(\tau)d\tau = \int \pi_\theta(\tau) \nabla_\theta\log\pi_\theta(\tau)r(\tau)d\tau = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \nabla_\theta\log\pi_\theta(\tau)r(\tau) \right]
$$

우리는 trajectory 확률 전체의 gradient가 아닌,  **trajectory를 구성하는 행동 하나하나의 확률 (정책)** 에 대한 gradient를 계산해야 한다. 그 이유는 trajectory는 policy와 환경이 만든 결과일 뿐이고, 우리가 **직접 조정할 수 있는 것은 policy뿐이기 때문이다.** 따라서 $\pi_\theta(\tau)$를 통해 $\nabla_\theta\log\pi_\theta(\tau)$를 재구성하면 다음과 같다.
$$
\pi_\theta(\tau) = p(s_1)\prod_{t=1}^T\pi_\theta(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)
$$
이므로
$$
\log \pi_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T\left(\pi_\theta(a_t\mid s_t)+p(s_{t+1}\mid s_t,a_t)\right)
$$
따라서 이에 대한 gradient를 구하면 
$$
\nabla_\theta\log\pi_\theta(\tau) = \nabla_\theta\left[\log p(s_1) + \sum_{t=1}^T\left(\pi_\theta(a_t\mid s_t)+p(s_{t+1}\mid s_t,a_t)\right) \right]
$$
인데 $\nabla_\theta\log p(s_1) = 0$ 이고 $p(s_{t+1} \mid s_t, a_t)$ 또한 $\theta$와 관련이 없기 때문에 $\theta$ 입장에서는 상수이다. 따라서, $\nabla_\theta \, p(s_{t+1} \mid s_t, a_t) = 0$ 이다.
따라서 $J(\theta)$에 대한 gradient는 다음과 같이 나타낼 수 있다.
$$
\nabla_\theta \, J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \nabla_\theta\log\pi_\theta(\tau)r(\tau) \right] = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\left(\sum_{t=1}^T\pi_\theta(a_t\mid s_t)\right) \left(\sum_{t=1}^T r(s_t,a_t)\right) \right]
$$
하지만 이때, 기댓값은 보통 직접 계산할 수 없다.  왜냐하면 가능한 trajectory의 경우의 수는 무수히 많기 때문이다. 특히 환경이 복잡하거나 상태 공간이 클 경우, 이러한 기댓값을 수학적으로 정확히 계산하는 것은 사실상 불가능에 가깝다.  따라서 우리는 **Monte Carlo Estimation** 기법을 사용하여 이를 근사하게 된다.  즉, 전체 trajectory 공간에서 **N** 개의 trajectory를 샘플링하고, 이를 통해 해당 gradient를 근사하는 방식이다.
$$
\nabla_\theta\,J(\theta) \approx \frac{1}{N}\sum_{i=1}^N\left(\sum_{t=1}^T\nabla_\theta\log\pi_{\theta}(a_{i,t}\mid s_{i,t})\right) \left(\sum_{t=1}^T r(s_{i,t}, a_{i,t})\right)
$$
이제 다음과 같이 이렇게 구한 $\nabla_\theta \, J(\theta)$를 사용하여 $\pi_\theta$에 대한 gradient ascent를 적용할 수 있고, 이를 반복적으로 수행하여 보상을 최대화할 수 있는 최적의 parameter $\theta^\ast$를 찾을 수 있다.
$$
\theta^\ast \leftarrow \theta + \alpha\nabla_\theta\,J(\theta)
$$
하지만 이 방식 또한 한계가 존재한다.
1. High-variance gradient
	$\rightarrow$ 샘플링을 통해 근사한 gradient는 **높은 분산(high variance)** 을 가지며, 이로 인해 학습이 불안정하고 수렴 속도가 느릴 수 있다
	- 이를 해결할 수 있는 방법에는 크게 두 가지가 존재한다
		- 매우 많은 sample을 추출한다.
		- 분산이 낮은 estimator를 사용한다.
	- 첫 번째 방법은 매우 많은 sample을 추출하기 때문에 그 비용이 매우 크고 이를 매번 gradient를 계산할 때마다 추출해야 하기 때문에 그 시간도 매우 오래 걸릴 것이다. 따라서 이 방법은 선호되지 않는다.
	- 두 번째 방법에서는 높은 분산의 문제점을 해결하기 위해 다음과 같은 3가지 estimator를 선택할 수 있다.
		1. Reward to-go : $\sum_{t^\prime=t}^T r(s_{t^\prime}^i + a_{t^\prime}^i)$  
		2. Better estimate of reward to-go : $Q_{\phi}^{\pi_\theta}(s_t^i,a_t^i)$
		3. Baseline: $V^{\pi^\theta}_\phi(s_t^i)$를 baseline으로 사용해 분산을 줄인다.
		4. Advantage: $A_\phi^{\pi_\theta}(s_t^i,a_t^i)=Q_\phi^{\pi_\theta}(s_t^i,a_t^i) - V_\phi^{\pi_\theta}(s_t^i)$
		5. Actor-critic algorithm: 정책(actor)과 가치 함수(critic)를 동시에 학습하여 분산을 줄이고 샘플 효율성을 높인다.
2. Require on-policy data
	$\rightarrow$ Policy Gradient는 **현재의 정책** $\pi_\theta$​ 에 의해 생성된 데이터(trajectory)만을 사용하여 학습할 수 있다.  즉, 과거에 다른 정책으로 수집한 데이터는 사용할 수 없기 때문에, **데이터 효율(sample efficiency)** 이 낮고 **매번 새로운 roll-out**이 필요하다
3. Local minimum problem
	$\rightarrow$ local optima에 빠져 전역 최적해를 찾기 어렵게 만들 수 있다.
## Reducing the variance of policy gradient
앞서 높은 분산으로 인한 문제점을 얘기했다. 이를 줄이기 위한 방법 3가지 Reward-to-go, Baseline, Advantage, Actor-critic algorithm을 알아가 보고자 한다.
### Reward to-go
기존 Policy Gradient 방식의 문제점 중 하나는 **전체 보상 $r(\tau)$** 를 고려한다는 점이다.  
하지만 생각해보면, 현재 시점 $t$에서의 행동 $a_t$는 이미 지나간 **과거의 보상에는 영향을 줄 수 없다.**  
그럼에도 불구하고 전체 trajectory의 총 보상을 모든 시점에 일괄적으로 적용하게 되면,  
이는 학습 과정에서 **불필요하게 높은 분산(high variance)** 을 유발할 수 있으며, 결과적으로 **정책 업데이트의 효율을 저하시킬 수 있다**.
이를 개선하기 위해 도입된 것이 바로 **Reward-to-go** 기법이다.  
이 방법은 각 시점 $t$에서의 policy gradient를 계산할 때, **그 시점 이후의 미래 보상만을 고려**한다.  
즉, $t$ 이후의 보상만 남기고 그 이전의 누적 보상은 제거함으로써, **보다 정확하고 안정적인 gradient 추정이 가능**해진다.

Reward-to-go는 수학적으로 다음과 같이 표현된다:
$$
\sum_{t=t^\prime}^T r(s_{t^\prime},a_{t^\prime})
$$
이를 써서 Policy gradient를 다음과 같이 바꿀 수 있다.
$$
\nabla_\theta\,J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( \sum_{t'=t}^{T} r(s_{t'}, a_{t'}) \right)
$$
이 방식은 기존 방식과는 다르게 전체 보상을 고려하지 않고 미래에 얻을 보상만 고려하여 variance를 줄이면도 unbiased한 gradient 추정이 가능하여 안정성을 높일 수 있다는 장점을 가진다.

### Value functions
가치함수는 현재 상태 $s_t$ 혹은 $(s_t,a_t)$로부터 앞으로의 **누적 보상의 기댓값**을 의미한다. 즉, 강화학습에서 agent가 어떤 상태나 행동이 얼마나 좋은가를 정량적으로 평가할 때 사용할 수 있는 지표이다. 에이전트는 이러한 가치 함수를 사용하여 더 나은 의사결정을 내릴 수 있게 된다.
1. State-Value
	정책 $\pi$를 따를는 agent가 상태 $s_t$일 때 앞으로 얻게 되는 보상의 기댓값이다. 이를 수식으로 나타내면 다음과 같다.
	$$
	V_{\phi}^{\pi_\theta}(s_t) = \sum_{t^{\prime} = t}^T\mathbb{E}_{\tau\sim\pi(\tau\mid s_t)} [r(s_{t^{\prime}},a_{t^{\prime}})]
	$$
	이는 사실 
	$$
	V_{\phi}^{\pi_\theta}(s_t) = \mathbb{E}_{a_i\sim\pi(a_t\mid s_t)} \left[Q_{\phi}^{\pi_\theta}(s_t, a_t) \right]
	$$
	로도 쓸 수 있다. 그 이유는 다음에 설명하도록 하자.
	또 Bellman-Equation 형태로 바꾸면 
	$$
	V_{\phi}^{\pi_\theta}(s_t)=E_{a_t\sim \pi(a_t, s_t), \,s_{t+1}\sim p(\cdot\, \mid s_t,a_t))}\left[r_t+V^{\pi}(s_{t+1})\right]
	$$
2. State-Action Value 
	State-Action Value function 혹은 Q-value는 action의 Quality를 평가하는 지표라고 할 수 있다. State value와는 달리 Action까지 고려하여 평가를 진행한다.  즉, 이는 상태 $s_t$에서 행동 $a_t$를 했을 때 앞으로 받을 누적 보상의 기댓값이다. 이를 수학적인 식으로 나타내면 다음과 같다.
	$$
	Q_\phi^{\pi_\theta}(s_t, a_t) =\sum_{t^{\prime}=t}^T \mathbb{E}_{\tau\sim\pi(\tau\mid s_t,a_t)}\left[r(s_{t^\prime}, a_{t^\prime})\right]
	$$
	이를 Bellman Equation 형태로 바꾸면 다음과 같다.
	$$
	Q_\phi^{\pi_\theta}(r_t, s_t) = \mathbb{E}_{s_{t+1} \sim p(\cdot \mid s_t, a_t)} 
\left[ r_t + \mathbb{E}_{a_{t+1} \sim \pi(\cdot \mid s_{t+1})} 
\left[ Q^\pi(s_{t+1}, a_{t+1}) \right] \right]
	$$
	상태 $s_t$에서 정책 $\pi$를 따를 때 얻을 것으로 기대되는 총 보상을 의미한다.
3. Advantage
	특정 행동이 평균보다 얼마나 좋은 지를 나타내는 측정량이다. 즉, 상태 $s_t$​에서 행동 $a_t$를 취했을 때,  **그 행동이 해당 상태에서 기대되는 평균보다 얼마나 나은지를 측정**한다.이를 수학적으로 나타내면 다음과 같다.
	$$
		A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)
	$$
	그러면 이 $A^\pi$가 좋은 지 안 좋은 지는 다음과 같이 판단할 수 있다.
	1. $A>0$ : 평균보다 좋은 행동
	2. $A<0$ : 평균보다 나쁜 행동
	하지만 이 식의 문제점이라고 하면 $Q-value$를 직접 계산하기 어렵다는 점이다. 그 이유는 현실적으로 전체 trajectory를 계산한다는 것은 불가능하기 때문이다. 따라서 우리는 bootstrap approximation 방법을 사용하여 $Q^\pi$를 $V^\pi$에 대해 근사할 수 있다. 
	그 식의 전개 과정은 다음과 같다. $$ Q^{\pi_\theta}_\phi(s_t^i, a_t^i) 
= \sum_{t'=t}^{T} \mathbb{E}_{\tau \sim \pi_\theta(\tau \mid s_t^i, a_t^i)} \left[ r(s_{t'}, a_{t'}) \right] \\
= r_t^i + \mathbb{E}_{s_{t+1} \sim p(s_{t+1} \mid s_t^i, a_t^i)} \left[ V^{\pi_\theta}(s_{t+1}) \right] \\
\approx r_t^i + V^{\pi_\theta}(s_{t+1}^i) \\
\approx r_t^i + V^{\pi_\theta}_\phi(s_{t+1}^i)$$ 
	따라서 Advantage를 다음과 같이 정의할 수 있다.
	$$
	A^{\pi_\theta}_\phi(s_t^i, a_t^i) 
	= r_t^i + V^{\pi_\theta}_\phi(s_{t+1}^i) - V^{\pi_\theta}_\phi(s_t^i)
	$$
	이제 우리가 Advantage 값을 개선시키기 위해 할 것은 $V^{\pi_\theta}_\phi(s)$를 $V^{\pi_\theta}(s)$에 fit하게 하는 것이다.즉, parameter를 조정하며 모델의 값을 실제 관측값에 가까워 지도록 학습시키는 것이다.
	이를 하기 위해서는 다음과 같은 방법을 사용할 수 있다.
	현재 ${(s_t, a_t, r_t)}$ 데이터를 가지고 있다고 하자. 우리의 target은 
	$$
	y_t^i = \sum_{t^\prime=t}^T \mathbb{E}_{\tau\sim\pi_\theta(\tau\mid s_t^i)}\left[r(s_{t^{\prime}}, a_{t^\prime})\right]
	$$
	이고 이를 근사시키면
	$$
	y_t^i \approx r_t^i + \sum_{t^{\prime}=t+1}^T \mathbb{E}_{\tau\sim\pi_\theta(\tau\mid s_{t+1}^i)}[r(s_{t^{\prime}}, a_{t^\prime})] \approx r_t^i + V_\phi^{\pi_\theta}(s_{t+1}^i)
				$$
	이고, $r_t^i + V_\phi^{\pi_\theta}(s_{t+1}^i)$ 를 **Temporal difference target**이라고 한다. (= TD target)
	이제 state value function을 fit하기 위해서는 목표값과 모델 값 간의 오차를 줄여야한다. 따라서 우리는 Loss function값을 사용하여 이 오차를 측정하고 이 오차를 줄이는 방향으로 fit할 것이다. 이 Loss를 여기서는 TD error라고 하고 식으로 표현하면 다음과 같다.
	$$
	\mathscr{L}(\phi) = \frac{1}{2}\sum_i||V_{\phi}^{\pi_\theta}(s^i) - y_i ||^2
	$$
	이렇게 측정한 오차를 바탕으로 우리는 학습하는 과정을 거쳐 다시 모델을 Loss를 줄이는 방향으로 학습시킨다. 이 과정을 TD Learning이라고 하며 이를 수식으로 나타내면 다음과 같다.
	$$
	V_\phi^{\pi_\theta}(s) \leftarrow r + V_\phi^{\pi_\theta}(s{^\prime})
	$$
	Advantage를 계산할 때도, Advantage의 식이
	$$
	A_\phi^{\pi_\theta}(s, a) = Q_\phi^{\pi_\theta}(s_t, a_t)-V_\phi^{\pi_\theta}(s_t)
	$$
	이고, 이전에 state-action value를 sdtate value로 나타내는 법을 알아보았는데 이를 사용하면 Advantage를 다음과 같이 나타낼 수 있었다.
	$$
		A^{\pi_\theta}_\phi(s_t^i, a_t^i) 
	= r_t^i + V^{\pi_\theta}_\phi(s_{t+1}^i) - V^{\pi_\theta}_\phi(s_t^i)
	$$
	이제 우리는 $V_\phi^{\pi_\theta}(s_t)$만 fit해주면 된다.
#### Discount factor
discount factor (할인 계수)는 미래 보상의 현재 가치를 결정하는 중요한 hyperparameter이다. 이는 지금 받는 보상이 나중에 받는 보상보다 더 가치 있다는 경제학적 원리를 반영한 것이다. 이 할인 계수를 사용하여 누적 보상을 수학적으로 나타내면 다음과 같다.
$$
\sum_{t=1}^T \gamma^{t-1}r_t
$$
따라서 만약 $\gamma$ 값이 클 수록 장기간 reward 최대화에 집중하며, 작아질 수록 agent는 근시안적이게 되어, 가까운 미래 보상에만 집중하게 된다. 따라서 $\gamma = 0$일 경우에는 즉각적인 보상만 고려하게 되고, $\gamma=1$일 경우에는 모든 미래 보상은 현재 보상과 다를 바 없이 동등하게 고려된다. 보통 할인율 값은 0.99로 두는데 그 이유는 만약 episode 길이 T가 무한대라면, 누적 보상 합이 발산할 것이기 때문이다. 
#### Better estimate of reward to go
이전에 살펴보았떤 reward-to-go 방식에서는 과거 보상은 고려하지 않고 미래에 얻게될 보상 만을 고려하여 누적 보상합을 구하였다. 이는 다음 식으로 나타낼 수 있었다.
$$
\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta\log\pi_\theta(a_t^i \mid s_t^i)\left(\sum_{t^\prime=t}^Tr(s_{t^\prime}^i,a_{t^\prime}^i ) \right)
$$
이때, $\sum_{t^\prime=t}^Tr(s_{t^\prime}^i,a_{t^\prime}^i )$이 부분을 Monte-Carlo 추정치 라고도 부르며 $\hat Q_t^i$로도 나타낸다. $\hat Q_t^i$는 한 trajectory마다 한 번만 사용하기 때문에 variance가 크다. 즉, $i=1 \sim N$ 까지 $N$개의 trajectory에 대한 기댓값을 $J(\theta)$의 gradient로 나타내고 있는데 샘플링 된 하나의 trajectory의 보상만 고려되고 다음 trajectory로 넘어가면 이전에 보았던 trajectory의 reward 정보는 고려되지 않는다. 이러한 이유로 variance가 높아지게 된다. 따라서 이 variance를 줄이기 위해 정책 하에 기대되는 미래 보상의 합, 기댓값으로 해당 값을 대체하면 된다.
$$
Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}_{\pi_\theta}\left[\sum_{t^\prime=t}^Tr(s_{t^\prime}, a_{t^\prime}) \mid s_t, a_t\right]
$$
#### Baseline
베이스라인은 정책 경사법에서 보상 신호에서 빼는 기준값이라고 할 수 있다. 이는 "상대적인 좋음"을 측정하기 위한 역할을 한다. 즉, 모든 행동의 보상에서 일정한 값을 빼줌으로써, 평균 이상으로 좋은 행동은 양의 값을, 평균 이하로 좋은 행동은 음의 값을 갖게 됩니다.
예를 들어, 학생들의 시험 점수를 생각해보면. 절대적인 점수보다는 평균 점수와의 차이(상대적 점수)가 학생의 성취도를 더 잘 나타낼 수 있다. 베이스라인은 이 "평균 점수"와 같은 역할을 하게된다.
Baseline은 수식에서 다음과 같이 추가되어 나타날 수 있다.
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta(\tau)} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left[ R_t(\tau)- b(s_t) \right]
$$
여기서 짚고 넘어갈 점은 $\pi_\theta(\tau)$와 $b$는 독립적이므로 $b$가 기댓값에 영향을 미치지 않는다. 그럼 이 baseline이 어떻게 분산에 영향을 미칠까?
variance 식을 기댓값으로 나타내면 다음과 같다.
$$
Var(X) = E[X^2] -E[X]^2
$$
이를 바탕으로 baseline을 적용하기 전 gradient를 바탕으로 식을 분산을 나타내면
$$
\operatorname{Var}(\nabla_\theta J(\theta)) = \operatorname{Var}\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) R_t(\tau)\right) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}\left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)R_t(\tau)\right)^2\right] - \mathbb{E}_{\tau \sim \pi_\theta(\tau)}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)R_t(\tau)\right]^2
$$
이다.
이것과 Baseline을 적용한 것과 비교하기 위해 Baseline을 적용한 분산 식을 구해보면
$$
\operatorname{Var}(\nabla_\theta J(\theta)) = \operatorname{Var}\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)(R_t(\tau) - b(\mathbf{s}_t))\right) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}\left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)(R_t(\tau) - b(\mathbf{s}_t))\right)^2\right] - \mathbb{E}_{\tau \sim \pi_\theta(\tau)}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)(R_t(\tau) - b(\mathbf{s}_t))\right]^2
$$
여기서 $E[X]^2$ 부분은 baseline이 있으나 없으나 같은 기댓값을 가지는데 $E[X^2]$ 값은 그렇지 않다.
$E[X^2]$을 보면.
$$
\mathbb{E}_{\tau \sim \pi_\theta(\tau)}\left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)R_t(\tau)\right)^2\right]
$$
이것과
$$
\mathbb{E}_{\tau \sim \pi_\theta(\tau)}\left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)(R_t(\tau) - b(\mathbf{s}_t))\right)^2\right]
$$
이것을 비교하면 된다.
여기서 만약에 $R_t(\tau) - b(s_t) \approx 0$이면 분산이 줄어들 수 있다. 따라서 $b(s_t)$가 보상 $R_t$를 가장 잘 추론해야 되고, baseline은 $s_t$ 값에만 의존하므로 state value를 baseline으로 사용할 수 있다.
따라서 Policy gradient with baseline을 다음과 같이 표현할 수 있다.
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta(\tau)} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left[ R_t(\tau)- V^{\pi_\theta}(s_t) \right]
$$
이렇게 함으로써, 에피소드 전체 보상 중에서 해당 상태 $s_t$ 상태에서 기대되는 평균 보상만큼을 상쇄시켜 그 시점에서의 초과 보상(advantage)만큼만 반영하게 된다. 결과적으로 variance가 줄어들게 된다. 여기서 Q-value를 better reward to go estimate로 사용하게 되면 다음 식으로 나타낼 수 있다.
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta(\tau)} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left[ Q^{\pi_\theta}(s_t, a_t)- V^{\pi_\theta}(s_t) \right]
$$
여기서 $Q^{\pi_\theta}(s_t, a_t)- V^{\pi_\theta}(s_t)$ 는 Advantage $A^{\pi_\theta}(s_t,a_t)$이고, 이는 Actor-critic algorithm에서 사용하는 형태와 같다. 즉 식을 다시 정리하면 다음과 같다.
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta(\tau)} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) A^{\pi_\theta}(s_t,a_t)
$$
따라서 value function을 이용해 policy gradient를 수행하면 학습을 위해 모든 trajectory를 수집하지 않아도 되므로 효율적으로 sampling할 수 있고 variance를 줄일 수 있는 효과가 있다. Actor-critic algorithm은 정책 네트워크(Actor)와 가치 네트워크(critic)을 함께 학습하며, Critic이 TD error를 최소화하며 상태 가치를 학습하면, Actor는 이 값을 기반으로 Advantage를 계산하여 정책을 업데이트하게 된다. 이렇게 하면 여러 trajectory  전부를 수집할 때마다 policy를 update하는 REINFORCE와는 다르게 trajectory 진행 중 정해진 N-step 마다 $V_\phi^{\pi_\theta}, \, Q_\phi^{\pi_\theta}$를 update하고 이를 통해 Actor가 $A^{\pi_\theta}_\phi(s,a)$.
