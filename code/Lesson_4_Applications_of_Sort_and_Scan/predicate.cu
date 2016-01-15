// Create predicate array for HW4

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * Calculate if LSB is 0.
 * 1 if true, 0 otherwise.
 */
__global__
void predicate_kernel(unsigned int *d_predicate,
                      unsigned int *d_val_src,
                      const size_t numElems) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;
  d_predicate[mid] = (int)((d_val_src[mid] & 1) == 0);
}

int main(void) {
  size_t numElems = 16;
  int ARRAY_BYTES = sizeof(unsigned int) * numElems;

  // device memory
  unsigned int* d_val_src;
  unsigned int* d_predicate;
  checkCudaErrors(cudaMalloc((void **) &d_val_src,   ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate, ARRAY_BYTES));

  // input array
  unsigned int h_input[numElems] = {39,21,84,40,78,85,13,4,47,45,91,60,74,8,44,53};
  checkCudaErrors(cudaMemcpy(d_val_src, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice));

  // kernel call
  predicate_kernel<<<1,16>>>(d_predicate, d_val_src, numElems);

  // debugging
  unsigned int *h_predicate = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_predicate, d_predicate,  ARRAY_BYTES, cudaMemcpyDeviceToHost));

  printf("INPUT \t PRED \n");
  for (int i = 0; i < numElems; i++)
    printf("%u \t %u\n", h_input[i], h_predicate[i]);

  return 0;
}
