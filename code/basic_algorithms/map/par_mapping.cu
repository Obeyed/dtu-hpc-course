#include <par_mapping.h>

__global__ 
void mapping_kernel(unsigned int* const d_out, 
                    unsigned int* const d_in, 
                    const size_t NUM_ELEMS) {
	const unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId >= NUM_ELEMS) return;
	d_out[myId] = d_in[myId];
}

void par_map(unsigned int* const h_out, 
             unsigned int* const h_in,
             const size_t NUM_ELEMS) {
  const int BLOCK_SIZE  = 512;
  const int GRID_SIZE   = NUM_ELEMS / BLOCK_SIZE + 1;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * NUM_ELEMS;

  // device memory
  unsigned int *d_in, *d_out;
  checkCudaErrors(cudaMalloc((void **) &d_in, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_out,   ARRAY_BYTES));

  // Transfer the arrays to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
  // run kernel
  mapping_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, NUM_ELEMS);
  // copy values to host
  cudaMemcpy(h_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost);
  // free device memory
  cudaFree(d_in); cudaFree(d_out);
}
