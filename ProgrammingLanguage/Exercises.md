### 8.4 - structure type
(a) x와 z는 structural equivalent하지만 x와 y는 같은 structure data type을 갖고 x와 z는 서로 다른 structure data type이므로
### 8.5 - array equivalence
array 변수는 첫 번째 element의 주소를 가지고 있기 때문에 x와 y는 각각 선언될 때 서로 다른 메모리 주소에 할당되므로 x에 y를 할당하면 compiler error가 발생한다.
pointer 변수를 활용하면 이 compile error를 제거할 수 있다.
### 8.6  - enumerated type
C 에서는 enum은 그저 변수에 주어진 alias이다. 따라서 type-checking은 이루어지지 않는다. 따라서 어떤 값에도 assignment는 가능하다. 반면 C++에서는 변수에 주어진 alias임은 같지만 strongly type checking이 이루어진다. 따라서 assignment하면 error가 발생한다.
### 8.12 - subtype and derived type
- Subtype
	- It is not a new type, it is just a subpart of the main type
	- allows applying range constraints on the main type
	- No conversion is required as it is a subset of the main type
- Derived type
	- It is a new type, derived from the parent type
	- allows range checking on parent type only but not on its derived version
	- Explicit conversion is required to make it compatible with other types along with parent type

### 10.27 - dangling reference
(a)
A stack-based environment is suitable for a block-structured language but it has some limitations as well, one such limitation is the dangling reference issue. Whenever a procedure is returning a pointer to a local object, it will result in a dangling reference. The activation record associated with the procedure will be deallocated automatically from the stack. The given scenario can be checked statically by using a symbol table, which records each entry of a variable whether it is a local or a global variable. Hence, the return statement or assignment statement can be checked using the symbol table.

각 변수의 entry가 local인지 global인지 저장하는 symbol table을 이용해 정적으로 checking할 수 있다. 따라서 return 문이나 assignment문은 symbol table을 이용해 check할 수 있다.