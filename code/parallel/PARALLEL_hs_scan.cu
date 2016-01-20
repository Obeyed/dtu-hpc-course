#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

int set_grid(int SIZE, int BLOCK_SIZE)
{
  return SIZE/BLOCK_SIZE + ((SIZE % BLOCK_SIZE)? 1 : 0);
}

// Performs one step of the hillis and steele algorithm for integers
__global__ void hs_kernel_global(int *d_out, int *d_in, int step, int SIZE) {
	// setting ID
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// checking if out-of-bounds
	if (tid >= SIZE) return;
	// setting itself
	int val = d_in[tid];
	// finding the number to add, checking out-of-bounds
	int toAdd = (((tid - step) < 0) ? 0 : d_in[tid - step]);
	// setting output
	d_out[tid] = val + toAdd;
}

void hs_kernel_wrapper(int * d_out, int * d_in, int SIZE, unsigned int BYTES, int NUM_THREADS) {
	// initializing starting variables
	int NUM_BLOCKS = SIZE/NUM_THREADS + 1;
	// initializing and allocating an "intermediate" value so we don't have to change anything in d_in
	int *d_intermediate;
	cudaMalloc((void **) &d_intermediate, BYTES);
	cudaMemcpy(d_intermediate, d_in, BYTES, cudaMemcpyDeviceToDevice);

  // stops when step is larger than array size, happens at O(log2(SIZE))
  for (int step = 1; step < SIZE; step <<= 1) {
		hs_kernel_global<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_intermediate, step, SIZE);
		cudaMemcpy(d_intermediate, d_out, BYTES, cudaMemcpyDeviceToDevice);
	}
	cudaFree(d_intermediate);
}


/* -------- MAIN -------- */
int main(int argc, char **argv)
{
  std::ofstream myfile;
  myfile.open ("par_scan.csv");
  // Setting NUM_THREADS
  const unsigned int times = 10;
  for (unsigned int rounds = 0; rounds<30; rounds++)
  {
//    printf("Round: %d\n", rounds);
    int NUM_THREADS = 1<<10;
    // Making non-bogus data and setting it on the GPU
    int SIZE = 1<<rounds;
    unsigned int BYTES = SIZE * sizeof(int);
    int * d_in;
    int * d_out;
    cudaMalloc(&d_in, sizeof(int)*SIZE);
    cudaMalloc(&d_out, sizeof(int)*SIZE);
    int * h_in = (int *)malloc(SIZE*sizeof(int));
    int * h_out = (int *)malloc(SIZE*sizeof(int));
    for (unsigned int i = 0; i <  SIZE; i++) h_in[i] = 1;
    cudaMemcpy(d_in, h_in, sizeof(int)*SIZE, cudaMemcpyHostToDevice);

    // Running kernel wrapper
    // setting up time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel time!!!
    cudaEventRecord(start, 0);

    for (unsigned int i = 0; i < times; i++)
    {
      hs_kernel_wrapper(d_out, d_in, SIZE, BYTES, NUM_THREADS);

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calculating time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime = elapsedTime / ((float) times);
//    printf("time!: %.5f\n", elapsedTime);
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
//    printf("%d \n", h_out);
    myfile << elapsedTime << "," << std::endl;
  }
  myfile.close();
  return 0;

}
