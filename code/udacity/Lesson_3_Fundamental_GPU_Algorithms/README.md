# Notes for Lesson 3
My own notes for lesson 3
## Algorithms
### Step complexity
How many steps in the computational graph? (I.E. critical path diagram)
### Work complexity
How many tasks (nodes) of work?
## Reduce
1. Inputs: Array of elements
2. Reduction operator
  a. Binary operation (a + b -> out)
  b. Associative a + (b + c) -> (a + b) + c
3. Reduce[(19,8,9,14), +] -> 50
4. Associative operations counts: Multiply, Minimum, Logical or, bitwise And
### Implementation of reduce - Serial
for i in range, sum += sum + i;
### Implementation of reduce - Parallel
Take (((a+b)+c)+d -> (a+b)+(c+d) - opposite tree structure
## Scan
Calculating running sum of input
1. Input: Array of elements(like reduce)
2. Binary associative operator(like reduce)
3. Identity element [I op a = a]
### Exclusive scan
All elements before, NOT including current element.
### Inclusive scan
All elements before, including current element.
### Hillis/Stefle inclusive scan
