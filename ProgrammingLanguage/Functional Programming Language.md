#### Functional Programming
1. unifrom view of programs as functions
2. Treats functions as data
3. Prevention of side effects
4. simpler semantics, simpler model for computation $\rightarrow$ rapid prototyping, ai...

#### Function
- Partial function : f is not defined for all x in X
- Total function : f is defined for all x in X

#### imperative vs functional
$\rightarrow$ concept of variable
* Imperative : refer to memory location that store values
* functional : variable always stand for actual values $\rightarrow$ no concepts of memory location and assignment $\rightarrow$ bound to values 하지만 대부분 assignment의 개념을 도입하고 있다.
functional languages have an advantage for *concurrent applications*

#### Composition 
- essential operation on functions

#### LISP
- lambda calculus
- single general structure : *list*
- *metacircular interpreter*
- *Automatic memory management*

#### Element of scheme
- Atoms
- Parenthesized expression
- Syntax is expressed in *Backus-Naur form*
- *prefix form*
- Evaluation rule represents *applicative order evaluation*
	- _All subexpressions are evaluated first_
	- quote : no evaluation
- Scheme function application use *pass by value*
- Special forms in scheme and lisp use *delayed evaluation*
- *let* (specail form) : provides local environment and scope
	- *temporary variable declarations*
- Lambda special form : creates function with specified formal params and a body of code evaluated when function is applied

#### Higher-Order functions
- take other functions as parameters and functions that return functions as values
- *Garbage collection* : automatic memory management technique to return memory used by functions

#### ML(Meta Language)
* avoids use of many parentheses
* statically typed
* More secure
* Highly efficient by type-checking at compile time

#### Haskell
* Pure functional language
* *monads*