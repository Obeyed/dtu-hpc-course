#include "par_reduce.h"

// Computes the sum of elements in d_in
__global__ 
void reduce_kernel(unsigned int* const d_out,
                   unsigned int* const d_in,
                   const size_t NUM_ELEMS) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
    if ((tid < s) && ((pos + s) < NUM_ELEMS))
      d_in[pos] = d_in[pos] + d_in[pos + s];
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid == 0) && (pos < NUM_ELEMS))
    d_out[blockIdx.x] = d_in[pos];
}

// Calls reduce kernel to compute reduction.
void reduce_wrapper(unsigned int* const d_out,
                    unsigned int* const d_in,
                    size_t num_elems,
                    int block_size) {
  unsigned int grid_size = num_elems / block_size + 1;

  unsigned int* d_tmp;
  checkCudaErrors(cudaMalloc(&d_tmp, sizeof(unsigned int) * grid_size));
  checkCudaErrors(cudaMemset(d_tmp, 0, sizeof(unsigned int) * grid_size));

  unsigned int prev_grid_size;
  unsigned int remainder = 0;
  // recursively solving, will run approximately log base block_size times.
  do {
    reduce_kernel<<<grid_size, block_size>>>(d_tmp, d_in, num_elems);

    remainder = num_elems % block_size;
    num_elems  = num_elems / block_size + remainder;

    // updating input to intermediate
    checkCudaErrors(cudaMemcpy(d_in, d_tmp, sizeof(int) * grid_size, cudaMemcpyDeviceToDevice));

    // Updating grid_size to reflect how many blocks we now want to compute on
    prev_grid_size = grid_size;
    grid_size = num_elems / block_size + 1;      

    // updating intermediate
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaMalloc(&d_tmp, sizeof(int) * grid_size));
  } while(num_elems > block_size);

  // computing rest
  reduce_kernel<<<1, num_elems>>>(d_out, d_in, prev_grid_size);
}

void par_reduce(unsigned int* const h_out, 
             unsigned int* const h_in,
             const size_t NUM_ELEMS) {
  const int BLOCK_SIZE  = 512;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * NUM_ELEMS;

  // device memory
  unsigned int *d_in, *d_out;
  checkCudaErrors(cudaMalloc((void **) &d_in, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_out,   ARRAY_BYTES));

  // Transfer the arrays to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
  // run kernel
  reduce_wrapper(d_out, d_in, NUM_ELEMS, BLOCK_SIZE);
  // copy values to host
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  // free device memory
  cudaFree(d_in); cudaFree(d_out);
}