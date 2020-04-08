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

### Files

###### **normal_vec-add.cpp**

This file is the array addition example in normal C++. It is there for 
comparison against the CUDA version.

###### **cuda_vec-add.cu**

This file is the array addition example using CUDA code. This example uses 
the manual copying of memory using the `cudaMemcpy()` calls, with no usage
of the unified memory.

###### **cuda_vec-add-unified.cu**

This is the array addition example in CUDA using the unified memory. 

###### **benchmark.txt**

This file shows the output of the Linux time command when run with 100M items.
The benchmark was generated using the `normal_vec-add.cpp` program and the
`cuda_vec-add-unified.cu` example.
