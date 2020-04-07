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
void add(int n, float *x, float *y)
{
  // Get the index for the thread that
  // is running this kernel
  int currentThread = threadIdx.x;

  // Get the size of a block
  int blockSize = blockDim.x;

  // Loop over each thread. You start with
  // the current thread, and then jump
  // by the size of a block. This means
  // that each thread handles 1 / blockSize
  // of the data.
  for (int i = currentThread; i < n; i += blockSize)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 100<<20; // 100M elements

  float *x, *y;

  // Allocate memory using Unified memory. 
  // Accessible either from the device or host.
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add<<<1, 256>>>(N, x, y);

  // Wait for the GPU to finish before accessing the host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free the memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
