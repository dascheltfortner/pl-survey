/**
 *  This is an example of using CUDA C to transform an 
 *  image from normal colors to grayscale. It uses the
 *  STB image library to load in the image.
 *
 *  The CUDA kernel comes from the book Programming 
 *  Massively Parallel Processors, 3rd Ed. (ch. 2)
 *
 *  https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands-ebook/dp/B01NCENHQQ
 */

#include <cuda.h>
#include <iostream>

 // You have to define these macros before you
 // define the stb headers per the docs
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

__global__
void colorToGrayscaleKernel(unsigned char* d_out, unsigned char* d_in, 
                            int w, int h, int nChannels) {

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if(col < w && row < h) {
    
    // Get 1D coordinate for the grayscale image
    int grayOffset = row * w + col;

    int rgbOffset = grayOffset * nChannels;

    unsigned char r = d_in[rgbOffset    ];
    unsigned char g = d_in[rgbOffset + 2];
    unsigned char b = d_in[rgbOffset + 3];

    d_out[grayOffset    ] = r;//0.21f*r;
    d_out[grayOffset + 1] = g;//0.71f*g;
    d_out[grayOffset + 2] = b;//0.07f*b;
  }

}

void colorToGrayscale(unsigned char* h_out, unsigned char* h_data,
                      int w, int h, int nChannels) {
  unsigned char* d_data;
  unsigned char* d_out;

  cudaMalloc(&d_data, sizeof(h_data));
  cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, sizeof(h_data));

  dim3 gridDimensions(ceil(w / 16.0), ceil(h / 16.0), 1);
  dim3 blockDimensions(16, 16, 1);

  colorToGrayscaleKernel<<<gridDimensions, blockDimensions>>>(d_out, d_data, w, h, nChannels);

  cudaMemcpy(h_out, d_out, sizeof(h_data), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  cudaFree(d_out);
}

int main() {

  const char* filename = "./test-img.jpg";
  int width, height, nChannels;

  unsigned char* h_data = stbi_load(filename, &width, &height, &nChannels, 0);
  
  unsigned char* result = (unsigned char*) malloc(sizeof(h_data));

  std::cout << nChannels <<std::endl;

  colorToGrayscale(result, h_data, width, height, nChannels);

  int len = sizeof(result) / sizeof(unsigned char);
  for(int i = 0; i < len; i++) {
    std::cout << (int)result[i] << " ";
  }

  std::cout << std::endl;

  const char* outputFile = "./gray.jpg";
  stbi_write_jpg(outputFile, width, height, nChannels, result, 100);

  stbi_image_free(h_data);
  free(result);
  
  return 0;

}
