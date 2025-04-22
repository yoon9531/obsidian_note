#### Reasons of Successful Language
1. Pragmatics
2. Killer applications
3. Uniformity of design concepts

#### Criteria of successful language
1. Achieves the goal of its designers
2. it attains *widespread use* in an application area
3. it serves as a model for other languages that are success ful

### 언어 설계의 초창기
$\rightarrow$ 기계들이 느리고 메모리는 부족하기 때문에 *Program Speed & memory usage*가 최우선 고려사항
* Efficiency : FORTRAN $\rightarrow$ directly mapped to machine code, minimizing amount of translation required by compiler
* Writability : less important than efficiency, 
	* ALGOL : block structure, structured control statements, structured array type, recursion
	* COBOL : attempt to imporve readability of programs
### 1970s~1980s
$\rightarrow$ *Simplicity & abstraction*
**Strong data type**
### 1980s~1990s
*logical or mathematical precision*
$\rightarrow$ **functional language**
#### 25년 간 가장 영향력있는 설계 기준
1. object-oriented approach to abstraction
2. use of libraries, reusability

#### Efficiency
1. FORTRAN, C/C++
2. *Strong data typing* : does not need to check data types before executing operations
3. FORTRAN : compile time에 모든 데이터 선언과 subroutine 호출이 알려져 있어야 함.(메모리 할당)
4. 실행 효율에 도움을 주는 features
	1. Static datatype : 효율적인 메모리 할당 or 접근을 할 수 있게 해준다.
	2. Manual memory management : overhead of garbage collection을 피하게 해준다.
	3. Simple sementics : simple structure of running programs
5. *Programmers Efficiency* 
	1. Conciseness of the syntax
6. *Expressiveness* : how easy to express
7. Reliability of program
	1. how easy errors can be found and corrected
	2. **maintainability**
#### Regularity
$\rightarrow$ How well the features of a language are integrated
1. Greater regularity
	1. Fewer restriction
	2. Fewer strange interaction
	3. Fewer surprises $\rightarrow$ principle of less astonishment
2. Concepts
	1. Generality : avoiding special cases
	2. Orthogonal design : constructs can be combined in any meaningful way, do not behave differently in different context
		1. Java : value semantics, reference semantics
		2. Smalltalk, Python : reference semantics only
	3. Uniformity : a design in which similar things look similar, consistency of appearance and behavior of language constructs
 3. 