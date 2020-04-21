/**
 *
 *  This is an example of a simple C++ program to add together two arrays.
 *  This comes from the Easy introduction to cuda from the Developer's 
 *  website:
 *
 *  https://devblogs.nvidia.com/even-easier-introduction-cuda/
 *
 * */

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *dst, float *src)
{
  for (int i = 0; i < n; i++)
      dst[i] = src[i] + src[i];
}

int main(void)
{
  int N = 100<<20; // 100M elements

  float *src = new float[N];
  float *dst = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    src[i] = 1.0f;
    dst[i] = 2.0f;
  }

  // Run kernel on N elements on the CPU
  add(N, dst, src);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] src;
  delete [] dst;

  return 0;
}
