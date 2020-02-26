## CUDA Research

#### Emulating CUDA Without NVidia Card

It appears that emulating NVidia cards without the hardware is very challenging.
There is a piece of softare called [ocelot](https://github.com/gtcasl/gpuocelot)
which will let you emulate CUDA code without an NVidia card. Unfortunately, the
last update to this repo was 5 years ago. A quick look into 
[installing](http://www.ieap.uni-kiel.de/et/people/kruse/tutorials/cuda/tutorial01o/web01o/tutorial01o.html)
Ocelot is a bit daunting. The big problem with Ocelot is that it's just really
old. CUDA is currently on version 10.x, but Ocelot, according to the tutorial 
I found, only supports CUDA up to version 4.0. This is a bit of a problem. 

There are a couple of tools that you can use to convert CUDA to OpenCl, thus
allowing you to run on a computer with AMD cards. This seems like a more viable
option than emulating. It looks like 
[CU2CL](http://chrec.cs.vt.edu/cu2cl/index.php) has good support and seems to 
do well. It has endorsements from the DoD, AMD, and a couple other big names.
There are a couple of other tools that do this same thing. AMD has released a 
way to port CUDA code to C++. This seems less straightforward than CU2CL, but
it will do the trick as advertised.
