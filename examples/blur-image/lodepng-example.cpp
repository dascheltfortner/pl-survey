/**
 *  This file uses a mix of the examples given in the lodepng/examples 
 *  folder. It simply loads a png image into a buffer, then it prints out
 *  an ascii representation of the image to the console.
 *
 *  It's really cool. You can use this example file to create new CUDA 
 *  examples that read in image data.
 *
 *  Lodepng is a cool library, and you can check the lodepng folder for 
 *  the information about this library. 
 *
 *  You can also find out more about lodepng here:
 *
 *  Github: https://github.com/lvandeve/lodepng
 *  Website: https://lodev.org/lodepng/
 * */
#include <iostream>
#include "lodepng/lodepng.h"

/*
Show ASCII art preview of the image
*/
void displayAsciiArt(const unsigned char* image, unsigned w, unsigned h) {
  if(w > 0 && h > 0) {
    std::cout << std::endl << "ASCII Art Preview: " << std::endl;
    unsigned w2 = 48;
    if(w < w2) w2 = w;
    unsigned h2 = h * w2 / w;
    h2 = (h2 * 2) / 3; //compensate for non-square characters in terminal
    if(h2 > (w2 * 2)) h2 = w2 * 2; //avoid too large output

    std::cout << '+';
    for(unsigned x = 0; x < w2; x++) std::cout << '-';
    std::cout << '+' << std::endl;
    for(unsigned y = 0; y < h2; y++) {
      std::cout << "|";
      for(unsigned x = 0; x < w2; x++) {
        unsigned x2 = x * w / w2;
        unsigned y2 = y * h / h2;
        int r = image[y2 * w * 4 + x2 * 4 + 0];
        int g = image[y2 * w * 4 + x2 * 4 + 1];
        int b = image[y2 * w * 4 + x2 * 4 + 2];
        int a = image[y2 * w * 4 + x2 * 4 + 3];
        int lightness = ((r + g + b) / 3) * a / 255;
        int min = (r < g && r < b) ? r : (g < b ? g : b);
        int max = (r > g && r > b) ? r : (g > b ? g : b);
        int saturation = max - min;
        int letter = 'i'; //i for grey, or r,y,g,c,b,m for colors
        if(saturation > 32) {
          int h = lightness >= (min + max) / 2;
          if(h) letter = (min == r ? 'c' : (min == g ? 'm' : 'y'));
          else letter = (max == r ? 'r' : (max == g ? 'g' : 'b'));
        }
        int symbol = ' ';
        if(lightness > 224) symbol = '@';
        else if(lightness > 128) symbol = letter - 32;
        else if(lightness > 32) symbol = letter;
        else if(lightness > 16) symbol = '.';
        std::cout << (char)symbol;
      }
      std::cout << "|";
      std::cout << std::endl;
    }
    std::cout << '+';
    for(unsigned x = 0; x < w2; x++) std::cout << '-';
    std::cout << '+' << std::endl;
  }
}

int main() {
  unsigned error;
  unsigned char* image = 0;
  unsigned width, height;

  const char* filename = "./test-img.png";

  error = lodepng_decode32_file(&image, &width, &height, filename);
  if(error) {
    printf("error %u: %s\n", error, lodepng_error_text(error));
  }

  displayAsciiArt(image, width, height);
  std::cout << std::endl;

  /*
    This code shows how you would encode a file using lodepng:

    unsigned error = lodepng_encode32_file(filename, image, width, height);

    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  */

  free(image);

  return 0;

}
