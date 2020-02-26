## CUDA Research

#### Source Materials

There are a couple of materials that would be good to pull information from
for the slides. 

* [Operating Systems](https://amzn.to/2SYYpNI)
I recall that this textbook had good things to say GPUs and the difference 
between CPUs and GPUs. This would be an excellent source for describing the 
difference between the CPU and GPU to lay the foundation for what CUDA is doing,
and why it's different.

* [CUDA by Example](https://amzn.to/2I0lXLY)
This book is published by NVIDIA. Thus, the authority of the book is pretty 
high. It does a good job of introducing the architecture of CUDA and it has
several examples on how to do things in CUDA. This is probably a good resource
to heavily leverage for the slides. This book is available on Safari 
[here](https://bit.ly/2Iap1Fr).

* [Programming Massively Parallel Processors](https://amzn.to/2Prm5YV)
This book is very similar to *CUDA by Example*. It has good examples at the end
though, with a case study of different applications. This could be good for 
thinking about fun homework problems. This book is available on Safari 
[here](https://bit.ly/391jkVP). 

* [Professional CUDA C Programming](https://bit.ly/2HWxWtP)
I haven't looked much into this book, but I put it here in case it has 
standard things. This might be a good reference to see if example CUDA code
for the slides or homework answers is good and up to par.

* [The CUDA Handbook: A Comprehensive Guide to GPGPU Programming](https://bit.ly/2w46l7s)
This book looks like it would be an invaluable resource for understanding what
is going on with the hardware behind GPGPU programming. I think that this
book will be really helpful for understanding GPU programming so as to 
understand what the CUDA code is doing on the hardware.

#### CUDA Documentation Links

These are useful links to CUDA stuff. 

* [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
This quick start guide covers installation on various platforms.

* [CUDA C++ Programming Guide](https://bit.ly/3860pYB)
This is a reference for C++ and CUDA. It covers some of the background. This 
is a good primer on the topic.

* [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
The CUDA Runtime API docs.

* [CUDA Toolkit Docs](https://docs.nvidia.com/cuda/archive/8.0/)
These are the docs for the CUDA Toolkit. 

## OpenCL Research

Book sources:

* [OpenCL by Example](https://learning.oreilly.com/library/view/opencl-programming-by/9781849692342/)
This is a lot like the *CUDA by Example* book, but for OpenCL.

* [OpenCL Programming Guide](https://learning.oreilly.com/library/view/opencl-programming-guide/9780132488006/)
This book contains a lot of information for writing OpenCL code. It also has 
some "case studies" with example applications. This could be useful for 
potential homework.

* [OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/opencl-2.0.pdf)
This is a very technical document that defines the OpenCL specification.

* [OpenCL References](https://www.khronos.org/opencl/)
This is the references put out by Khronos about OpenCL. There don't appear to 
be easy docs, but there are some helfpul materials here.

Note that OpenCL is a *standard*, it's not a *language*. So, OpenCL code has
to be supported by a graphics card, and then you use their implementation for
your code. So, if you, for example, wanted to use NVIDIA's OpenCL 
implementation, you would still need a NVIDIA card. 
