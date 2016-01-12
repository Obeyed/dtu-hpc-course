#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream>

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
	int NUM_BLOCKS = SIZE/NUM_THREADS + ((SIZE % NUM_THREADS)?1:0);
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

int main(int argc, char **argv) {
  std::ofstream myfile;
  myfile.open ("par_scan.csv");

  int NUM_THREADS = 1 << 10,
      SIZE,
      TIMES = 10,
      MAX = 29;
  unsigned int BYTES;
  int *h_in, *h_out,
      *d_in, *d_out;

  // MAXIMUM SIZE: ALLOCATE ONCE!
  BYTES = MAX * sizeof(int);
  // setting host memory
  h_in  = (int *)malloc(BYTES); 
  h_out = (int *)malloc(BYTES);
  // allocate GPU memory
  cudaMalloc((void **) &d_in, BYTES);
  cudaMalloc((void **) &d_out, BYTES);

	for (int rounds = 0; rounds <= MAX; rounds++) {
    SIZE = 1 << rounds;
    BYTES = SIZE * sizeof(int);

		for(int i = 0; i < SIZE; i++)
      h_in[i] = 1;

		// transfer arrays to GPU
		cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);

    // setting up time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
		// kernel time!!!
		for (int i = 0; i < TIMES; i++)
	    hs_kernel_wrapper(d_out, d_in, SIZE, BYTES, NUM_THREADS);

    // stop recording
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // calculating time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime = elapsedTime / ((float) TIMES);
    printf("average time elapsed: %f\n", elapsedTime);

    myfile << elapsedTime << ",";
	}
  // move last result back to host
  cudaMemcpy(h_out, d_out, BYTES, cudaMemcpyDeviceToHost);
  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);

  myfile.close();

  for (int i = 0; i < 5; i++)
    printf("%d ", h_out[i]);
  printf(" -- ");
  for (int i = SIZE - 5; i < SIZE; i++)
    printf("%d ", h_out[i]);
  printf("\n");

	return 0;
}
