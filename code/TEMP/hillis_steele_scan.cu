#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>

void sequential_solution(int *, int *, const int, const unsigned int, const unsigned int, const unsigned int);
void kernel_test_module(int *, int *, const int, const unsigned int, const unsigned int, const unsigned int);

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

void kernel_test_module(int * h_out, int * h_in, const int num_times, const unsigned int ARRAY_SIZE, const unsigned int ARRAY_BYTES, const unsigned int num_threads)
{
	// setting device pointers
	int * d_in;
	int * d_out;
	printf("defined device pointers successfully... \n");

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);
	printf("malloc device pointers successfully... \n");

	// transfer arrays to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemset(d_out, 0, ARRAY_BYTES);
	printf("copy device pointers successfully... \n");

	int counter = 0;
	while(counter < num_times)
	{	
		std::cout << "john";
		hs_kernel_wrapper(d_out, d_in, ARRAY_SIZE, ARRAY_BYTES, num_threads);
		counter++;
	}
	// back to host
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
}

int main(int argc, char **argv)
{
	/* -------- Initialization -------- */
	printf("Hillis and Steele ONLINE... \n");

	// defining vars
	const unsigned int num_threads = 1<<9;
	const unsigned int ARRAY_SIZE = 1<<24;
	const int num_times = 1;
	const unsigned int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	printf("defined vars... \n");
	printf("ARRAY_SIZE: %d\n", ARRAY_SIZE);
	printf("ARRAY_BYTES: %d\n", ARRAY_BYTES);

	/* -------- TESTING TIME! -------- */
	std::vector<void (*)(int *, int *, const int, const unsigned int, const unsigned int, const unsigned int)> funcs;

	funcs.push_back(kernel_test_module);
	funcs.push_back(sequential_solution);


	std::vector<char *> names;
	names.push_back("PARALLEL");
	names.push_back("SEQUENTIAL");




	for (int i = 0; i < funcs.size(); i++)
	{
		// setting host in
		int * h_in  = (int *)malloc(ARRAY_BYTES); // allocates to memory
		int * h_out = (int *)malloc(ARRAY_BYTES);

		for(int i = 0; i < ARRAY_SIZE; i++)
		{
			h_in[i] = 3;
			h_out[i] = 0;
		}
		std::cout << "filled array..."<< std::endl;
		std::cout << "Setting up for: " << names[i] << std::endl;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Compute time!
		(*funcs[i])(h_out, h_in, num_times, ARRAY_SIZE, ARRAY_BYTES, num_threads);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		
		// calculating time
		float elapsedTime;
    	cudaEventElapsedTime(&elapsedTime, start, stop);    
    	elapsedTime /= (float)num_times;
		
		// printing
		for(int i = 400; i<408; i++)
		{
			std::cout << "index " << i << ": count " << h_out[i] << std::endl;
		}

		std::cout << "average time elapsed: " << elapsedTime << std::endl;
		free(h_in);
		free(h_out);
	}


	return 0;
}

