### 1. Superscalar processors (SISD)
![[Pasted image 20250423003530.png]]
- Superscalar processor는 1개의 processor로 병렬성을 구현한다. 이 병렬성은 ILP를 통해 구현된다.(Instruction Level Parallelism)
- 위 그림에서 확인할 수 있듯이 superscalar processor는 두 개의 execution unit이 있다. 1개 이상의 명령어를 실행 유닛에 할당하여 1 clock cycle에 여러 개의 명령어를 실행할 수 있다. 이는 1개의 single processor에서 실행된다.
- Scalar processor : 1 cycle에 오직 하나의 명령어만 fetch/decode 될 수 있다.
#### ILP(Instruction Level parallelism)
다음 Assembly 코드는 `a=x*x + y*y + z*z`명령어를 assembly 코드로 나타낸 것이다.
```c
//r0=x, r1=y, r2=z, r3=a 
mul r0, r0, r0 
mul r1, r1, r1 
mul r2, r2, r2 
add r0, r0, r1 
add r3, r0, r2
```
이 코드는 위에서부터 순차적으로 실행된다. 하지만 x, y, z에 대한 각 계산은 서로에게 의존성이 없기 때문에 분리되어 계산할 수 있다. super scalar processor는 이러한 독립적인 명령어들을 자동으로 찾아내서 병렬적으로 실행한다. 이는 Dynamic scheduling을 통해 이루어지며 dynamic scheduling은 runtime에 명령어들의 순서를 변경할 수 있다.
하지만 위 명령어가 모두 independent하지 않다. 다음 다이어그램을 보면
![[Pasted image 20250423004203.png]]
(4)는 (1), (2)가 수행된 후에 계산되어야 한다. 이 경우 data dependency가 존재한다. 따라서 위 계산을 병렬적으로 실행해도 3번의 cycle이 지난 후에 연산이 완료될 수 있다. 하지만 이전 순차적으로 실행했을 때 5 cycle이 걸린 것에 비해 더 빠르게 실행된다고 할 수 있다.
이전에 super scalar processor가 자동으로 indendent한 명령어를 찾아서 병렬로 실행해준다고 했다. 말로만 들었을 때는 좋아보이지만 모든 프로그램에는 data dependency가 있기 마련이다. 때문에, 이 아키텍처도 한계가 존재한다.

### Multicore Processor (MIMD)
Multicore processor는 다음과 같이 여러 개의 core를 가지고 있다.
![[Pasted image 20250423004817.png]]
하지만 multicore processor에서 각 core는 single core processor보다 느리다. 예를 들어 멀티코어 프로세서의 각 코어가 싱글 코어의 80% clock frequency를 갖고 있다고 해보자. 각 core는 단일 core의 관점에서 single core processor보다 느린 것이 맞지만 multicore processor에서는 이 core를 여러개 갖고 있고 이 그림에서는 2개가 존재한다. 따라서 $2*0.8=1.6$이므로 single core processor보다 빠르다고 할 수 있다.
그럼 아까 위에서 봤던 연산을 멀티코어 프로세서에선 어떻게 실행될까? 그건 확실하게 답을 할 수 없다. 프로그래머가 직접 코드를 작성해야하고 여기서 사용하는 것이 쓰레드이다. 간단히 설명하자면 thread는 경량화된 프로세스로 프로세스 내의 실행 단위라고 할 수 있다. Process와 thread말고도 subroutine과 thread의 차이도 있는데 이를 먼저 살펴보면
- **Subroutine**: thread가 subroutine을 호출하면 여전히 하나의 thread control이 있지만 해당 control은 subroutine의 thread로 넘어간다. 여기서는 여전히 싱글 코어를 사용한다.
- **Thread**: 만약 thread가 다른 thread를 호출하면 호출한 thread의 제어권과는 별개로 새로운 제어권을 가진 thread가 생성된다. 이 thread들은 병렬적으로 실행가능하며, 호출한 thread가 종료돼도, 계속 실행될 수 있다. 이러한 특징을 가지고 멀티코어 프로세서는 쓰레드를 가지고 병렬화를 구현한다.
병렬 프로그래밍을 목적에 두고 공부하고 있으니 thread를 이해할 필요가 있다. thread는 프로세스의 경량화 버전으로 프로세스 내의 실행 단위이다. 따라서 이것 또한 scheduling의 대상이다. Linux에서는 이 thread를 PCB를 공유하는 정도를 구분하여 process와 thread를 구분한다. Thread는 더 많은 정보를 공유한다. thread 간에는 code, data, heap 등의 리소스를 공유한다. 하지만 그렇다고 모든 리소스를 공유하는 것은 아니고 **stack이나 PC, register 같은 리소스는 공유하지 않는다**. thread도 결국 실행 단위이기 때문에 지역 변수나 함수 리턴 값이 저장될 곳이 있어야겠고, scheduling할 때 어디 실행하고 있었는 지를 기록해야하기 때문에 PC 레지스터를 비롯한 여러 레지스터가 필요하다. 이러한 리소스들은 thread 독립적으로 가지고 있어야 한다. 따라서 이러한 리소스들은 공유되지 않는다. 

> [!note] Superscalar vs Multiprocessor
> - Superscalar : 여러 개의 명령어를 동시에 실행하지만 single core processor이기 때문에 프로그램 입장에서 single thread에서 실행된다.
> - Multicore processor : 두 개의 thread로 명령어를 실행한다.

다음은 multi thread 프로그램 실행의 한 예시다.
![[Pasted image 20250423010328.png |500]]
2개의 코어가 각 연산의 절반씩을 할당하여 수행하고 있다. 물론 싱글 코어로 수행할 때보다 더 효율적으로 연산을 수행하는 것은 맞지만 모든 코어에서 fetch/decode 과정을 수행하고 있다. 하지만 사이클마다 데이터만 다르고 같은 명령어를 실행하기 때문에 명령어를 가져오는 건 한 번만 해도 된다. 따라서 fetch/decode unit이 두 개나 있을 필요가 없다. 이러한 문제를 해결하기 위해 여기서 vector processing이 등장한다.
### Vector processing
여기서는 하나의 fetch/decode unit만 존재하기 때문에 하나의 instruction stream만 있게 되고 여러 개의 ALU가 하나의 fetch/decode unit을 공유한다.
![[Pasted image 20250423010944.png]]
각 여러 개의 ALU는 서로 다른 여러 개의 데이터를 가지고 연산을 수행하게 된다.  
이 processor에서 조건문은 어떻게 실행될까?
![[Pasted image 20250423011458.png]]
이 그림에 나와있는 것과 같이 masking 혹은 predication을 통해 condition에 따라 어떤 ALU가 실행될 지 결정한다. compiler가 masking vector를 만들어서 조건에 맞는 ALU를 실행하고 다른 거는 다 꺼버린다. 하지만 당연히 모든 ALU를 fully utilize하지 못하기 때문에 효율성 문제가 있다. 
