#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Performs one step of the hillis and steele algorithm for integers
__global__ void hs_kernel_global(int * d_out, int * d_in, int step, const unsigned int ARRAY_SIZE)
{
	// setting ID
	int myId = threadIdx.x + blockDim.x * blockIdx.x;

	// checking if out-of-bounds
	if(myId >= ARRAY_SIZE)
	{
		return;
	}

	// setting itself
	int myVal = d_in[myId];

	// finding the number to add, checking out-of-bounds
	int myAdd;
	if((myId - step)<0)
	{
		myAdd = 0;
	}
	else
	{
		myAdd = d_in[myId-step];
	}

	// setting output
	d_out[myId] = myVal + myAdd;
}

void hs_kernel_wrapper(int * d_out, int * d_in, const unsigned int ARRAY_SIZE, const unsigned int ARRAY_BYTES, const unsigned int num_threads)
{
	// initializing starting variables
	unsigned int num_blocks = ((ARRAY_SIZE) / num_threads) + 1;
	int step = 1;
	// initializing and allocating an "intermediate" value so we don't have to change anything in d_in
	int * d_intermediate;
	cudaMalloc((void **) &d_intermediate, ARRAY_BYTES);
	cudaMemcpy(d_intermediate, d_in, ARRAY_BYTES, cudaMemcpyDeviceToDevice);
	int i = 1;
	while(step<ARRAY_SIZE) // stops when step is larger than array size, happens at O(log2(ARRAY_SIZE))
	{
		// for debugging purposes
		// printf("round %d: step %d\n", i, step);
		// i++;
		// one step/kernel at a time to do synchronization across blocks
		hs_kernel_global<<<num_blocks, num_threads>>>(d_out, d_intermediate, step, ARRAY_SIZE);
		cudaMemcpy(d_intermediate, d_out, ARRAY_BYTES, cudaMemcpyDeviceToDevice);
		step <<= 1; // double step size at each iteration

	}
	cudaFree(d_intermediate);
}

int main(int argc, char **argv)
{
	printf("Hillis and Steele ONLINE... \n");
	// defining vars
	const unsigned int num_threads = 512;
	const unsigned int ARRAY_SIZE = 1<<24;
	const unsigned int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned int);
	printf("defined vars... \n");
	printf("ARRAY_SIZE: %d\n", ARRAY_SIZE);

	// setting host in
	int * h_in  = (int *)malloc(ARRAY_BYTES); // allocates to memory
	int * h_out = (int *)malloc(ARRAY_BYTES);
	for(int i = 0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = 3;
		h_out[i] = 0;
	}
	printf("filled array... \n");

	// setting device pointers
	int * d_in;
	int * d_out;
	printf("defined device pointers... \n");

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);
	printf("malloc device pointers... \n");

	// transfer arrays to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	printf("copy device pointers... \n");

	// setting up time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// kernel time!!!
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; i++)
    {
    	hs_kernel_wrapper(d_out, d_in, ARRAY_SIZE, ARRAY_BYTES, num_threads);
    }
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// calculating time
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;

	// back to host
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// printing
	for(int i = 400; i<408; i++)
	{
		printf("index %d: count %d\n", i, h_out[i]);
	}

	printf("average time elapsed: %f\n", elapsedTime);

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}