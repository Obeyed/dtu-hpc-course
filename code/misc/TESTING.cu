#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

__global__ void test_overhead(unsigned int * d_out, unsigned int * d_in, const unsigned int SIZE){}

__global__ void test_overhead_init(unsigned int * d_out, unsigned int * d_in, const unsigned int SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;
	if (myId>=SIZE)
	{
		return;
	}
}

__global__ void test_read_global_coel(unsigned int * d_out, unsigned int * d_in, const unsigned int SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;
	if (myId>=SIZE)
	{
		return;
	}
	unsigned int global = d_in[myId];
}

__global__ void test_read_global_noncoel(unsigned int * d_out, unsigned int * d_in, const unsigned int SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;
	if (myId>=SIZE)
	{
		return;
	}
	unsigned int myVal = d_in[(myId % (SIZE/128))*128];
}

__global__ void test_write_global(unsigned int * d_out, unsigned int * d_in, const unsigned int SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;
	if (myId>=(1<<29))
	{
		return;
	}
	d_out[myId] = myId;
}

__global__ void test_write_shared(unsigned int * d_out, unsigned int * d_in, const unsigned int SIZE)
{
	extern __shared__ int sdata[];

	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;
	if (myId>=(1<<29))
	{
		return;
	}
	sdata[tid] = myId;
}

int main(int argc, char **argv)
{
	const unsigned int times = 1000;
	const unsigned int IN_SIZE = 1<<29;
	unsigned int IN_BYTES = sizeof(unsigned int) * IN_SIZE;
	unsigned int OUT_SIZE = 1<<29;
	unsigned int OUT_BYTES = sizeof(unsigned int) * OUT_SIZE;
	dim3 NUM_THREADS(1<<10); 
	dim3 NUM_BLOCKS(IN_SIZE/NUM_THREADS.x + ((IN_SIZE % NUM_THREADS.x)?1:0));

	// Generate the input array on host
	unsigned int * h_in = (unsigned int *)malloc(IN_BYTES);
	unsigned int * h_out = (unsigned int *)malloc(OUT_BYTES);
    for (unsigned int j = 0; j<IN_SIZE; j++) {h_in[j] = j;}

	// Declare GPU memory pointers
	unsigned int * d_in;
	unsigned int * d_out;

	// Allocate GPU memory
	cudaMalloc(&d_in, IN_BYTES);
	cudaMalloc(&d_out, OUT_BYTES);

	// Transfer the arrays to the GPU
	cudaMemcpy(d_in, h_in, IN_BYTES, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// Launching kernel
	for(int t = 0; t<times; t++)
	{
		//test_overhead<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
		//test_overhead_init<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
		//test_read_global_noncoel<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
		test_read_global_coel<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
	    //test_write_global<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
	    //test_write_shared<<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS.x*sizeof(unsigned int)>>>(d_out, d_in, IN_SIZE);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	// calculating time
	float elapsedTime = .0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapsedTime = elapsedTime / ((float) times);
	printf(" time: %.5f\n", elapsedTime);

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}