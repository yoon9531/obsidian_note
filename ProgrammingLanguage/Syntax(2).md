#### Ambiguity
- If sentence has more than two parse trees
- left recursive & right recursive

#### Imposing Assoiciativity
- can be imposed by controlling the recursion of nonterminals
#### Imposing precedence
- *precedence cascade* 
- Imposing the precedence of * over +
- Anoter solutions?
	- Fully-parenthesized expressions
	- Prefix expressions
* Scheme이 prefix expressions를 사용하면서도 괄호를 필요로 하는 이유?
	* Scheme allows any number of arguments to the arithmetic operators

#### Extended BNF
* [] : optional structures
* {} : repetitive structures
* Expressive power of EBNF is same to BNF
#### Syntax Diagrams
- Ovals : terminal
- Rectangular :  non-terminal
- Arrows : sequencing
- *Rarely seen anymore* : EBNF is more compact
#### Parsing
- Analyzing the syntax of program
- Construct parse tree
- *Top down parsing* : recursive-descent parsing, LL parsing
	- May cause infinite recursive loop
	- No way to decide which of the two choices to take until a+ is seen
	- term $\rightarrow$ term {+ term} (curly brackets in EBN represent left recursion removal)
	- expr $\rightarrow$ term @ expr | term
	- expr $\rightarrow$ term \[@ expr\]
	- *left factoring*
- *Bottom up parsing* : shift reducing parsing, reduce it to the nonterminal on the left, *preferred*
#### Parsing Problems
* Single symbol lookahead : using a single token to direct a parse
* Predictive parser
#### Scanner/Parser Generator
* scanner generator : produce scanner code from the given *regular expressions*
* parser generator : produces a parser code from the given *grammar specification file*
#### Lexics / Syntax / Semantics
- no clear cut between lexics and syntax
- Reserved words cannot be used as identifiers
- Predefined identifiers can be redefined in a program