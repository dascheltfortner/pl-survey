/**
 *
 *  This is a cuda version of the array addition program as created from the 
 *  tutorial from here:
 *
 *  https://devblogs.nvidia.com/even-easier-introduction-cuda/
 *
 *  Any adjustments made are made from suggestions from Programming Massively
 *  Parallel Processors, 3rd Edition:
 *
 *  https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0128119861/ref=dp_ob_title_bk
 *
 * */

#include <cuda.h>
#include <iostream>
#include <math.h>

            // Signifies a kernel function
__global__  // Runs on device code
void deviceAdd(int n, float *dst, float *src)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if(index < n) {
    dst[index] = src[index] + dst[index];
  }
}

void add(int n, float* h_dst, float* h_src) {
  int size = n * sizeof(float);
  float *d_dst, *d_src;
  
  // This allocates memory and copies 
  // the memory from the host to the 
  // device memory.
  cudaMalloc((void **) &d_src, size);
  cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_dst, size);
  cudaMemcpy(d_dst, h_dst, size, cudaMemcpyHostToDevice);

  deviceAdd<<<ceil(n / 256.0), 256>>>(n, d_dst, d_src);

  cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_src);
  cudaFree(d_dst);
}

int main(void)
{
  int N = 100<<20; // 100M elements

  float* x = (float*) malloc(N * sizeof(float));
  float* y = (float*) malloc(N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  free(x);
  free(y);

  return 0;
}
