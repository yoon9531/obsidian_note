## Recap
- Off-line RL
	- Learn from static data
	- Reusing data
	- When collecting data is expensive
	- Difficulty : handle unseen action
		- test this policy real world -> 어떻게 될 지 모른다.
	- Challenging : Q-value overestimation
		- Behavior policy : $\pi_\beta$
		- learned policy (predicting actions): $\pi_\theta$
		- 이거 두 개 다르다.

### TD3 + BC

### Conservative Q-Learning
- CQL (OOD actions들에 대해서는 높은 값 가질 수 없도록)
- Dataset에 없는 action들에 대해서는 평가하지 않기
- Predicted Q value가 data support 구간에서 true Q-value보다 작아지는 것을 방지하기 위해 이 데이터에 대해 push up
- Penalization + Bonus
- Imitate good trajectories(Filtered BC)
	- Advantage Weighted Regression(AWR)
		- $\hat V^{\pi_\beta}$ 
		- $\hat \pi$ 

#### Expectile Regression

### IQL : Implicit Q-Learning
- Converge $\rightarrow$ AWR 사용해서 $\pi$ 추출



> [!example] sdafkjlkj
> sdfjlkasf
> asdlkfjask
> dsalfkas
> dasflk
- 

- Hello im test
- dasfjlaskdfj

