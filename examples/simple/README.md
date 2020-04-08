# Simple Array Addition Example

The Array Addition is a good example of how to use CUDA. It's simple, kind of 
like a Hello World for parallel programming. 

This example demonstrates the following principles:

* Data parallelism
* Kernel/Device signifiers (`__global__`, etc)
* How to use threads and blocks in CUDA, along with the pre-defined variables 
    for them. (`<<<gridSize, blockSize>>>`, `threadIdX`, etc)
* Show the usage of memory copying between host and device, and show the 
    newer unified memory model.
* Show performance differences between the two

