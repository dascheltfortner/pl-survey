## Outline of Module Lectures

#### Lecture 1: GPU Overview

Goals of this lecture:

* Help the students understand the context for parallel programming, why it's 
necessary, and what problem it's trying to solve. Also show why CPUs are useful
alongside GPUs.

* Give an overview of the history of GPUs. This gives the necessary background
to understand the differences between things like OpenCL, DirectX, OpenGL, and
other models.

* Describe the differences in architecture between a GPU and a CPU. This is 
essential for helping the student understand how their CUDA C programs should
be written.

* Introduce CUDA, CUDA C, and some other implementations.

#### Lectures 2 - 3 (4?): CUDA C Examples

Goals of these lectures:

Demonstrate the core functionality of CUDA C. CUDA C is meant to be a minimal 
extension on C, which means there aren't many crazy new things to look at.
These lectures should demonstrate the use of:

* Kernel definition macros such as `__global__`, `__device__`, etc.

* Data parallelism/task parallelism

* Spawing kernels with the `<<< >>>` syntax.

* The pre-defined CUDA C variables that denote block and grid position such
as `threadIdX`, `blockIdX`, etc.

* 

#### Lecture(s) (4 &?) 5: Performance Considerations for CUDA

There are a lot of different considerations when writing CUDA code. Writing the
parallelized code is relatively simple, but actually getting it to run well
is a whole other ball game. When you choose what block and grid size to use, 
you have to make sure that what you chose actually gives you a performance 
boost. ([PMPP](https://bit.ly/2PXocnt), ch 5) You also need to consider the 
warp size. ([PMPP](https://bit.ly/2PXocnt), ch 5) In addition to those 
considerations, you need to think about the balance of memory and threads, and
how to allocate memory to minimize the traffic to and from global memory.
([PMPP](https://bit.ly/2PXocnt), ch 6) You also need to consider warp sizes
with thread divergence. Seemingly just the way you structure your if-statement
can have effects on performance. ([PMPP](https://bit.ly/2PXocnt), ch 6)
