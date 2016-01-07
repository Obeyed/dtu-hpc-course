# Notes for Lesson 2
My own notes for lesson 2
## Parallel communication patterns
### Map
Read & write to specific data elements, one-to-one correspondance.
### Gather
Read & write such that a thread accesses several data elements and may intersect with other threads while accessing and writes to one distinct element. Many-to-one correspondance.
### Scatter
Read & write such that a thread accesses one distinct element and writes to several, maybe intersecting, other elements. One-to-many
### Stencil - 2D von neumann(cross), 2D more(grid)
Read input from a fixed neighborhood in an array. LOTS of data reuse! Several-to-one
### Transpose
Transpose to re-order memory layouts. One-to-one

Array og structures(AoS), structure of arrays(SoA).
### How to efficiently access memory in concert
how to exploit data reuse. communicating partial results by sharing memory.
## GPU Hardware
### The programming model
- Divide program into kernels - C/C++ funtions
- threads, path of executions, different threads might take different paths.
- thread block is a group of threads that cooperate to solve a (sub)problem.
### Thread blocks and GPU hardware
- Why divide problem into blocks?

- When are blocks runned?
  - Randomly assigned to a computational unit
- In what order do they run
  - Randomly
- How can they cooporate
  - with what limitations?
### Cuda guarantees
- All threads are runned on the same SM at the same time
- All blocks in a kernel finish before blocks from the next kernel are launched
### memory model
- threads have local memory
- blocks have memory shared between threads
- all blocks on one GPU have access to the global GPU memory.
- the CPU has host memory, ursually information is copied between host and global GPU memory
### synchronization
- threads can access each other's results through shared and global memory
  - deadlock?
#### Barrier
- all the threads stop and wait until everyone has completed computing.
### Writing efficient programs
#### High-level strategies
- 3 TFLOPS per GPU
- First strategy
1. Maximize the arithmetic intensity Math/Memory
  2. Maximize work per thread
  3. Minimize time spent on memory per thread
#### Minimize time spent on memory
1. Move freq. accessed data to fast memory(local(registrers, l1 cashe) > shared >> global >> host)
#### Coalesced global memory access
1. When reading host memory, reads a large chunk. Use the following information of the chunk in other threads to minimize IO's
2. Stided is not as good
#### 10,000 reading SAME memory cell
1. it will read old values and store it in the new ones. RANDOM RESULT!!
2. atomic Add() atomic min()
3. atomix CAS
#### Limitations of atomic
1. Only certain operations, data types are accepted(mostly integer types)
2. workaround using CAS (comparing swap operation)
3. Still no ordering constraint floating arithmetics is non-associative
4. Serialize access to memory(slow!)
