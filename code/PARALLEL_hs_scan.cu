#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

// Performs one step of the hillis and steele algorithm for integers
__global__ void hs_kernel_global(unsigned int *d_out, unsigned int *d_in, int step, unsigned int SIZE) {
	// setting ID
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// checking if out-of-bounds
	if (tid >= SIZE) return;
	// setting itself
	unsigned int val = d_in[tid];
	// finding the number to add, checking out-of-bounds
	unsigned int toAdd = (((tid - step) < 0) ? 0 : d_in[tid - step]);
	// setting output
	d_out[tid] = val + toAdd;
}

void hs_kernel_wrapper(unsigned int * d_out, unsigned int * d_in, unsigned int SIZE, unsigned int BYTES, unsigned int NUM_THREADS) {
	// initializing starting variables
	unsigned int NUM_BLOCKS = SIZE/NUM_THREADS + ((SIZE % NUM_THREADS)?1:0);
	int step = 1;
	// initializing and allocating an "intermediate" value so we don't have to change anything in d_in
	unsigned int * d_intermediate;
	cudaMalloc((void **) &d_intermediate, BYTES);
	cudaMemcpy(d_intermediate, d_in, BYTES, cudaMemcpyDeviceToDevice);
	while(step<SIZE) // stops when step is larger than array size, happens at O(log2(SIZE))
	{
		hs_kernel_global<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_intermediate, step, SIZE);
		cudaMemcpy(d_intermediate, d_out, BYTES, cudaMemcpyDeviceToDevice);
		step <<= 1; // double step size at each iteration

	}
	cudaFree(d_intermediate);
}

int main(int argc, char **argv)
{
	std::ofstream myfile;
    myfile.open ("par_scan.csv");
	printf("Hillis and Steele ONLINE... \n");
	unsigned int times = 10;
	for (int rounds = 0; rounds<30; rounds++)
	{
		// defining vars
		unsigned int NUM_THREADS = 1<<10;
		unsigned int SIZE = 1<<rounds;
		unsigned int BYTES = SIZE * sizeof(unsigned int);
		printf("num: %d\n", rounds);		
		printf("SIZE: %d\n", SIZE);

		// setting host in
		unsigned int * h_in  = (unsigned int *)malloc(BYTES); // allocates to memory
		unsigned int * h_out = (unsigned int *)malloc(BYTES);
		for(unsigned int i = 0; i < SIZE; i++){h_in[i] = 1;}

		// setting device pointers
		unsigned int * d_in;
		unsigned int * d_out;

		// allocate GPU memory
		cudaMalloc((void **) &d_in, BYTES);
		cudaMalloc((void **) &d_out, BYTES);

		// transfer arrays to GPU
		cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);

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

		// back to host
		cudaMemcpy(h_out, d_out, BYTES, cudaMemcpyDeviceToHost);

		printf("average time elapsed: %f\n", elapsedTime);

		// free GPU memory allocation
		cudaFree(d_in);
		cudaFree(d_out);
        myfile << elapsedTime << ",";
	}
	myfile.close();
	return 0;
}
