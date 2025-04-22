#### Formal Language Hierarchy
1. Regular
2. Context-free
3. Context-Sensitive
4. Unrestricted

#### Syntax structure of programming language
1. *Lexical structures*
	- regular expressions
2. *Syntactic structures*
	- Context-free Grammar (CFG)
	- BNF(Backus-Naur Form) notation for CFG
#### Syntax
$\rightarrow$ structure of a language
- Variation of BNF
	- Original BNF
	- Extended BNF
	- Syntax Diagrams

#### Lexical structure
- tokens : words of programming languages
- *Scanning phase* : collects sequence of characters from input program
- *Parsing phase* : processes the tokens, determining programs syntactic structure

* *Regular expression*
	* used for text searching $\rightarrow$ *grep*
#### Token
- keywords
- Literals
- special symbols
- identifiers
- comments
- *Redefined identifier* : initial meaning given, but capable of redirection

#### Lexical Structure Issues
* *Token delimiters*
#### Regular Expressions
* Concatenation : done by sequencing the items
* Repetition : *
* choice, selection : indicated by a vertical bar
* [] : hyphen indicate a range of characters
* ? : indicates an optional item
#### Scanner
- Token recognizing program
	- Scanner
	- Scanner construction tool : build a scanner for a given scanner specification

#### Context-free Grammars
- grammar : a set of rewriting rules
- non-terminal : LHS'es of the arrow
- terminal : tokens
- start symbol : designated nonterminal

#### Parse Tree
- Syntax establishes *structure*
- Syntax-directed semantics
- All terminals and nonterminals in a derivation are included in the parse tree

#### Abstract Parse Tree
* Abstract the essential structure of the parse tree

#### Discussion Topics
* Why are Backus-Naur form and context-free grammars well-suited for describing the syntax of a programming language?
	* 지금 사용 -> 크게 3가지 구성요소만 있으면 다 표현할 수 있다. 이 3가지를 Backus-Naur form이 기술 가능하기 때문이다.