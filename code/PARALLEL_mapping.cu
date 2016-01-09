#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

__global__ void mapping_kernel(unsigned int * d_out, unsigned int * d_in, const unsigned int IN_SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	// checking for out-of-bounds
	if (myId>=IN_SIZE)
	{
		return;
	}
	d_out[myId] = d_in[myId];
}

int main(int argc, char **argv)
{
	std::ofstream myfile;
    myfile.open ("par_mapping.csv");
	printf("---STARTED---\n");
	const unsigned int times = 10;
	for(unsigned int rounds = 0; rounds<30; rounds++)
	{
		const unsigned int IN_SIZE = 1<<rounds;
		const unsigned int IN_BYTES = sizeof(unsigned int) * IN_SIZE;
		const unsigned int OUT_SIZE = IN_SIZE;
		const unsigned int OUT_BYTES = IN_BYTES;
		const dim3 NUM_THREADS(1<<10);
		const dim3 NUM_BLOCKS(IN_SIZE/NUM_THREADS.x + ((IN_SIZE % NUM_THREADS.x)?1:0));

		// Generate the input array on host
		unsigned int * h_in = (unsigned int *)malloc(IN_BYTES);
		unsigned int * h_out = (unsigned int *)malloc(OUT_BYTES);
	    for (unsigned int j = 0; j<IN_SIZE; j++) {h_in[j] = 1;} //printf("  h_in[%d]: %d\n", j, h_in[j]);}

		// Declare GPU memory pointers
		unsigned int * d_in;
		unsigned int * d_out;
	    printf("\n@@@ROUND@@@: %d\n", rounds);
	    printf("---IN_SIZE---: %d\n", IN_SIZE);
	    printf("---IN_BYTES---: %d\n", IN_BYTES);
	    printf("---OUT_SIZE---: %d\n", OUT_SIZE);
	    printf("---OUT_BYTES---: %d\n", OUT_BYTES);
	    printf("---THREAD_SIZE---: %d\n", NUM_THREADS.x);
	    printf("---NUM_BLOCKS---: %d\n", NUM_BLOCKS.x);

		// Allocate GPU memory
		cudaMalloc((void **) &d_in, IN_BYTES);
		printf("---ALLOCATED D_IN---\n");
		cudaMalloc((void **) &d_out, OUT_BYTES);
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
            mapping_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // calculating time
        float elapsedTime = .0f;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        elapsedTime = elapsedTime / ((float) times);
        printf(" time: %.5f\n", elapsedTime);

		// Copy back to HOST
		cudaMemcpy(h_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost);
		int sum = 0;
		for(unsigned int i = 0; i<OUT_SIZE; i++){sum += h_out[i];}
		for(unsigned int i = 0; (i<OUT_SIZE) && (i<10); i++)
		{
			printf("OUT %d: count %d\n", i, h_out[i]);
		}
		printf("%d\n", sum);
		// free GPU memory allocation
		cudaFree(d_in);
		cudaFree(d_out);
        myfile << elapsedTime << ",";
	}
	myfile.close();
	return 0;
}