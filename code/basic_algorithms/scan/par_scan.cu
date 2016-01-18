#include "par_scan.h"

// Performs one step of the Hillis and Steele algorithm for unsigned integers
__global__ 
void scan_kernel(unsigned int* const d_out, 
                 unsigned int* const d_in, 
                 int step, 
                 size_t SIZE) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= SIZE) return;

  int val = d_in[tid];
  int toAdd = (((tid - step) < 0) ? 0 : d_in[tid - step]);
  d_out[tid] = val + toAdd;
}

void scan_wrapper(unsigned int* const d_out, 
                  unsigned int* const d_in, 
                  const size_t SIZE, 
                  const unsigned int BYTES,
                  const unsigned int BLOCK_SIZE) {
  int GRID_SIZE = SIZE/BLOCK_SIZE + 1;

  // device memory
  unsigned int *d_tmp;
  cudaMalloc((void **) &d_tmp, BYTES);
  cudaMemcpy(d_tmp, d_in, BYTES, cudaMemcpyDeviceToDevice);

  // stops when step is larger than array size, happens at O(log2(SIZE))
  for (int step = 1; step < SIZE; step <<= 1) {
    scan_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_tmp, step, SIZE);
    cudaMemcpy(d_tmp, d_out, BYTES, cudaMemcpyDeviceToDevice);
  }
  cudaFree(d_tmp);
}

void par_scan(unsigned int* const h_out, 
              unsigned int* const h_in,
              const size_t NUM_ELEMS) {
  const int BLOCK_SIZE  = 512;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * NUM_ELEMS;

  // device memory
  unsigned int *d_in, *d_out;
  checkCudaErrors(cudaMalloc((void **) &d_in,  ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_out, ARRAY_BYTES));

  // Transfer the arrays to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
  // run kernel
  scan_wrapper(d_out, d_in, NUM_ELEMS, ARRAY_BYTES, BLOCK_SIZE);
  // copy values to host
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  // free device memory
  cudaFree(d_in); cudaFree(d_out);
}
