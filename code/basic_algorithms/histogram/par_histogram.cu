#include "par_histogram.h"

__global__ 
void histogram_kernel(unsigned int* const d_bins, 
                      unsigned int* const d_in, 
                      const unsigned int BIN_SIZE, 
                      const size_t IN_SIZE) {
	unsigned int mid = threadIdx.x + blockDim.x * blockIdx.x;
	if (mid >= IN_SIZE) return;

	unsigned int item = d_in[mid];
	unsigned int bin = item % BIN_SIZE;

	atomicAdd(&(d_bins[bin]), 1);
}


void par_histogram(unsigned int* const h_out, 
                   unsigned int* const h_in,
                   const size_t NUM_ELEMS,
                   const unsigned int BIN_SIZE) {
  const int BLOCK_SIZE  = 512;
  const int GRID_SIZE   = NUM_ELEMS / BLOCK_SIZE + 1;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * NUM_ELEMS;

  // device memory
  unsigned int *d_in, *d_out;
  checkCudaErrors(cudaMalloc((void **) &d_in,  ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_out, ARRAY_BYTES));

  // Transfer the arrays to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
  // run kernel
  histogram_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, BIN_SIZE, NUM_ELEMS);
  // copy values to host
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  // free device memory
  cudaFree(d_in); cudaFree(d_out);
}
