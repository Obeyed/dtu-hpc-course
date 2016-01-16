#include <stdio.h>

int NUM_BLOCKS = 4;
int BLOCK_WIDTH = 2;

__global__ void hello(){
  printf("Hello world! I'm %d thread in block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char **argv){

  // launch the kernel
  hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

  // force the printf()s to flush
  cudaDeviceSynchronize();

  printf("That's all!\n");

  return 0;
}
