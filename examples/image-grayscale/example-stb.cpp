/**
 *  This file is an example use of the STB library:
 *    
 *    https://github.com/nothings/stb
 *
 *  It is here to show the user of this example how STB
 *  is used in isolation from the CUDA implementations,
 *  before the blur example.
 *
 *  This file demonstrates how to load a file with STB,
 *  and how to write a file with STB. This example simply
 *  loads in a file, and then writes the same file to 
 *  a different location.
 */

#include <iostream>

// STB docs require you define the STB impelementation
// macro before you include the header
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

int main() {

  const char* filename = "./test-img.jpg";

  int x, y, n;

  // Load in the image. This sets the width to x, height to y,
  // and the n to the number of channels in the file. The final
  // parameter specifies the number of channels desired.
  unsigned char* imageData = stbi_load(filename, &x, &y, &n, 0);

  std::cout << "Loaded:" << std::endl;
  std::cout << "  Dimensions: " << x << "x" << y << std::endl;
  std::cout << "  unsigned char size: " << sizeof(unsigned char) << std::endl;
  std::cout << std::endl << "Writing to ./test-out.png" << std::endl;
  
  // Writes the image to an output file. The variables passed in 
  // are the same as they are set above. The final parameter, 
  // in this case, is the stride in bytes.
  stbi_write_png("./test-out.png", x, y, n, imageData, x * n);
  stbi_image_free(imageData);

  return 0;

}
