## GPU Lecture

Goals of this lecture:

- If I recall correctly, PL Survey is supposed to fit into the Sophomore year.
If that is true, then the students in this class either might or might have 
not taken Operating Systems. If they haven't taken OS, it will be good to have
a bit of an overview of the GPU and how GPU programming differs from CPU 
programming. If they have, a more detailed refresher might be pertinent. 

- Introduce the necessary hardware considerations addressed in 
[Programming Massively Parallel Processors](https://bit.ly/2PXocnt). The book
references several things about the GPU that would be critical information for
understanding parallel code on the GPU.

- Go over the history of GPU programming. 

- Introduce CUDA and explain what it is.

In citations, PMPP refers to [Programming Massively Parallel Processors](https://bit.ly/2PXocnt)

## GPU History & Architecture

- Discuss the need for GPUs, and the benefit over CPUs. The CPU design is for
sequential programs as described by vonNeumann. People expect faster processors
with each generation, but they are coming up on a natural barrier. They are now
adding more *cores* to CPUs, but the hardware isn't doing much better. So the
move is towards higher *parallelism*. Discuss the two types of parallelism.
(PMPP ch 1)

- GPUs and CPUs represent two trajectories to stretch performance. CPUs use the
*multicore* trajectory, GPUs use the *many-thread* trajectory. This is the key
difference between the two processors. GTX680 has 16,384 threads! (PMPP ch 1)

- Go over the history of the GPU. This is important for understanding CUDA and
what it is, and getting your head wrapped around the different concepts. There
is DirectX, OpenGL, OpenCL, CUDA and all of this stuff that gets very confusing.
All of these are probably things that the students have heard of. Putting them
all into their proper historical context is helpful. 

- Walk through the history of the GPU. Explain the fixed pipeline system that
was used for OpenGL and DirectX. During this era, GPUs were used only for 
shaders. Very briefly explain the pipeline. Next, GPUs allowed for 
programmable portions of the GPU. This meant that some of the pipeline steps
were programmable. Next, there was the GeForce 8800, followed by GPGPU.
Researchers who were eager to see the performance gains of using the GPU used
these pixel shaders to solve general problems. Finally, NVIDIA made some 
adjustments to their hardware and developed CUDA, which allows for fully 
programmable GPU programs. (PMPP ch 2)

- Give an overview of the GPU's architecture, and describe how this helps for 
performance for specific applications. Also give a crash course on the 
hardware concepts necessary to understand CUDA code later on. For example, it
is necessary to understand how the GPU uses things like blocks, grids, and warps
to write correctly optimized code. These concepts are described well in PMPP 
chapters 3 & 4. You also need to be mindful of the performance effects of 
using too many accesses to global memory. Since this is happening in a ton of
threads, you have not be wary of going to global memory too much. (PMPP chp 5)
Also give an overview of the differences between the traditional von Neumann
CPU design and the GPU design. (PMPP ch 5) 

- Introduce CUDA and OpenCL. This can be very brief, as it just shows that this
is the tookits that people use to program on the GPU. Also introduce at the 
end of this lecture what method the students will use to program the GPU.
