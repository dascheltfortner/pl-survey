/**
 *  This is an example of a more complicated kernel. It takes in an image
 *  and then runs an algorithm that blurs the picture slightly.
 *
 *  The CUDA kernel comes from the book Programming 
 *  Massively Parallel Processors, 3rd Ed. (ch. 2)
 *
 *  https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands-ebook/dp/B01NCENHQQ
 */

#include <cuda.h>
#include <iostream>

// Include the lodepng library
#include "lodepng/lodepng.h"

#define BLUR_SIZE 16

__global__
void blurKernel(unsigned char* d_out, unsigned char* d_in, int w, int h) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < w && row < h) {
    
    int pixelSum = 0;
    int numPixels = 0;

    for(int i = -BLUR_SIZE; i < BLUR_SIZE + 1; ++i) {
      for(int j = -BLUR_SIZE; j < BLUR_SIZE + 1; j++) {
        
        int currentRow = row + i;
        int currentCol = col + j;

        bool validRow = currentRow > -1 && currentRow < h;
        bool validCol = currentCol > -1 && currentCol < w;

        if(validRow && validCol) {
          pixelSum += d_in[currentRow * w + currentCol];
          numPixels++;
        }

      }
    }

    unsigned char pixelAverage = (unsigned char)(pixelSum / numPixels);
    d_out[row * w + col] = pixelAverage;
  }

}

int main() {

  std::string inFile = "./test-img.png";
  std::vector<unsigned char> image;
  unsigned width, height;

  unsigned decodeError = lodepng::decode(image, width, height, inFile);

  if(decodeError) {
    std::cout << "decoder error " << decodeError << ": " << lodepng_error_text(decodeError) << std::endl;
    return 1;
  }

  std::cout << image.size() << std::endl;

  unsigned char* d_image;
  unsigned char* d_blurred;

  cudaMallocManaged(&d_image, image.size() * sizeof(unsigned char) / 4);
  cudaMallocManaged(&d_blurred, image.size() * sizeof(unsigned char) / 4);

  dim3 gridDimensions(ceil(width / 32.0), ceil(height / 32.0), 1);
  dim3 blockDimensions(32, 32, 1);

  std::vector<unsigned char> result(image.size());

  for(int channel = 0; channel < 4; channel++) {
  
    for(int i = channel; i < image.size(); i += 4) {
      d_image[i / 4] = image[i];
    }

    blurKernel<<<gridDimensions, blockDimensions>>>(d_blurred, d_image, width, height);

    cudaDeviceSynchronize();

    for(int i = channel; i < image.size(); i += 4) {
      result[i] = d_blurred[i / 4];
    }
  }

  cudaFree(d_image);
  cudaFree(d_blurred);

  unsigned encodeError = lodepng::encode("./blur.png", result, width, height);
  
  if(encodeError) {
    std::cout << "encoder error " << encodeError << ": " << lodepng_error_text(encodeError) << std::endl;
    return 2;
  }


  return 0;

}
