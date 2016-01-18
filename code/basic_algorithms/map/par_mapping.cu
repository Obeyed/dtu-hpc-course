#include <par_map.h>
#include <cuda_runtime.h>

__global__ 
void mapping_kernel(unsigned int* const d_out, 
                    unsigned int* const d_in, 
                    const size_t NUM_ELEMS) {
	const unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId >= NUM_ELEMS) return;
	d_out[myId] = d_in[myId];
}

unsigned int* par_map(unsigned int* h_input, const size_t NUM_ELEMS) {
  const int BLOCK_SIZE  = 512;
  const int GRID_SIZE   = NUM_ELEMS / BLOCK_SIZE + 1;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * NUM_ELEMS;

  // host memory
  unsigned int* h_output = new unsigned int[NUM_ELEMS];

  // device memory
  unsigned int *d_input *d_out;
  checkCudaErrors(cudaMalloc((void **) &d_input, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_out,   ARRAY_BYTES));

  // Transfer the arrays to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
  // run kernel
  mapping_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, NUM_ELEMS);
  // copy values to host
  cudaMemcpy(h_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost);
  cudaFree(d_in); cudaFree(d_out);

  return h_out;
}
