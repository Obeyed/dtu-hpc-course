#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

__global__ void simple_histo(unsigned int * d_bins, unsigned int * d_in, unsigned int BIN_SIZE, unsigned int IN_SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	// checking for out-of-bounds
	if (myId>=(1<<29))
	{
		return;
	}
	unsigned int myItem = d_in[myId];
	unsigned int myBin = myItem % BIN_SIZE;
	atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void bad_histo(unsigned int * d_bins, unsigned int * d_in, unsigned int BIN_SIZE, unsigned int IN_SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	// checking for out-of-bounds
	if (myId>=IN_SIZE)
	{
		return;
	}
	unsigned int myItem = d_in[myId];
	unsigned int myBin = myItem % BIN_SIZE;
	d_bins[myBin]++;
}

int main(int argc, char **argv)
{
	std::ofstream myfile;
    myfile.open ("par_histogram.csv");
	printf("---STARTED---\n");
	unsigned int times = 10;
	// Vars
	unsigned int IN_SIZE;
	unsigned int IN_BYTES;
	unsigned int BIN_SIZE;
	unsigned int BIN_BYTES;
	unsigned int NUM_THREADS;
	unsigned int NUM_BLOCKS;
	unsigned int j;
	unsigned int sum;

	for(unsigned int rounds = 0; rounds<30; rounds++)
	{
		IN_SIZE = 1<<29;
		IN_BYTES = sizeof(unsigned int) * IN_SIZE;
		BIN_SIZE = 1<<rounds;
		BIN_BYTES = sizeof(unsigned int) * BIN_SIZE;
		NUM_THREADS = 1<<10;
		NUM_BLOCKS = IN_SIZE/NUM_THREADS + ((IN_SIZE % NUM_THREADS)?1:0);

		// Generate the input array on host
		unsigned int * h_in = (unsigned int *)malloc(IN_BYTES);
		unsigned int * h_bins = (unsigned int *)malloc(BIN_BYTES);
	    for (j = 0; j<IN_SIZE; j++) {h_in[j] = j;} //printf("  h_in[%d]: %d\n", j, h_in[j]);}

		// Declare GPU memory pointers
	    printf("\n@@@ROUND@@@: %d\n", rounds);
	    printf("---IN_SIZE---: %d\n", IN_SIZE);
	    printf("---IN_BYTES---: %d\n", IN_BYTES);
	    printf("---BIN_SIZE---: %d\n", BIN_SIZE);
	    printf("---BIN_BYTES---: %d\n", BIN_BYTES);
	    printf("---THREAD_SIZE---: %d\n", NUM_THREADS);
	    printf("---NUM_BLOCKS---: %d\n", NUM_BLOCKS);
	    unsigned * d_in;
		unsigned * d_bins;
		// Allocate GPU memory
		cudaMalloc(&d_in, IN_BYTES);
		printf("---ALLOCATED D_IN---\n");
		cudaMalloc(&d_bins, BIN_BYTES);
		printf("---ALLOCATED D_IN---\n");

		// Transfer the arrays to the GPU
		cudaMemcpy(d_in, h_in, IN_BYTES, cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // running the code on the GPU $times times        
        for (unsigned int k = 0; k<times; k++)
        {
            cudaMemset(d_bins, 0, BIN_BYTES);
            simple_histo<<<NUM_BLOCKS, NUM_THREADS>>>(d_bins, d_in, BIN_SIZE, IN_SIZE);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // calculating time
        float elapsedTime = .0f;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        elapsedTime = elapsedTime / ((float) times);
        printf(" time: %.5f\n", elapsedTime);

		// Copy back to HOST
		cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);
		sum = 0;
		for(unsigned int i = 0; i<BIN_SIZE; i++){sum += h_bins[i];}
		for(unsigned int i = 0; (i<BIN_SIZE) && (i<10); i++)
		{
			printf("bin %d: count %d\n", i, h_bins[i]);
		}
		printf("%d\n", sum);
		// free GPU memory allocation
		cudaFree(d_in);
		cudaFree(d_bins);
        free(h_in);
        free(h_bins);
        myfile << elapsedTime << ",";
	}
	myfile.close();
	return 0;
}