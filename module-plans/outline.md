## Outline of Module Lectures

#### Lecture 1: GPU Overview

###### Summary

I think it would be a good idea to get into the differences between 
GPUs and CPUs, and give a brief history lesson of GPU programming.
Programming in CUDA C without understanding the processor would be like
programming in normal C without understanding the processor: you could
do it, but you probably shouldn't. So, the first lecture of the module
should cover the architecture of a GPU and describe what makes 
parallel programming different and how GPU programming works at the level
of the GPU.

#### CUDA Performance

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

#### Computational Thinking?
