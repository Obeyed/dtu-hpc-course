#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT, int ARRAY_SIZE)
{
	unsigned int myId = threadIdx.x + blockDim.x + blockIdx.x;
	// checking for out-of-bounds
	if (myId>=ARRAY_SIZE)
	{
		return;
	}

	unsigned int myItem = d_in[myId];
	unsigned int myBin = min(static_cast<unsigned int>(BIN_COUNT - 1),
							 static_cast<unsigned int>(myItem % BIN_COUNT));
	atomicAdd(&(d_bins[myBin]), 1);
}

int main(int argc, char **argv)
{
	const int ARRAY_SIZE = 65536;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
	const int BIN_COUNT = 16;
	const int BIN_BYTES = BIN_COUNT * sizeof(int);

	// generate the input array on host
	int h_in[ARRAY_SIZE];
	for(int i = 0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = 3;
	}
	int h_bins[BIN_COUNT];
	for(int i = 0; i < BIN_COUNT; i++)
	{
		h_bins[i] = 0;
	}

	// declare GPU memory pointers
	int * d_in;
	int * d_bins;

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_bins, BIN_BYTES);

	// transfer the arrays to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

	// pop'in that kernel
	simple_histo<<<ARRAY_SIZE/64, 64>>>(d_bins, d_in, BIN_COUNT, ARRAY_SIZE);

	// copy back to HOST
	cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

	for(int i = 0; i<BIN_COUNT; i++)
	{
		printf("bin %d: count %d\n", i, h_bins[i]);
	}

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_bins);

	return 0;
}