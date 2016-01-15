// Create predicate array for HW4

#include "utils.h"
#include <stdio.h>
#include <vector>
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

__global__
void inclusive_sum_scan_kernel(unsigned int* d_sum_scan,
                               unsigned int* d_predicate,
                               int step,
                               const size_t numElems) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems)
    return;

	// finding the number to add, checking out-of-bounds
	int toAdd = (((mid - step) < 0) ? 0 : d_predicate[mid]);
  d_sum_scan[mid] = d_sum_scan[mid] + toAdd;
}

void DEBUG(unsigned int *device_array, unsigned int ARRAY_BYTES, size_t numElems) {
  unsigned int *h_test  = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_test, device_array, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  for (int i = 0; i < numElems; i++)
    printf("%u ", h_test[i]);
  printf("\n");
}

int main(void) {
  size_t numElems = 16;
  int ARRAY_BYTES = sizeof(unsigned int) * numElems;

  // device memory
  unsigned int *d_val_src, *d_predicate, *d_sum_scan;
  checkCudaErrors(cudaMalloc((void **) &d_val_src,   ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan,  ARRAY_BYTES));

  // input array
  unsigned int* h_input = new unsigned int[numElems];
  h_input[0]  = 39; h_input[1]  = 21; h_input[2]  = 84; h_input[3]  = 40;
  h_input[4]  = 78; h_input[5]  = 85; h_input[6]  = 13; h_input[7]  = 4;
  h_input[8]  = 47; h_input[9]  = 45; h_input[10] = 91; h_input[11] = 60;
  h_input[12] = 74; h_input[13] = 8;  h_input[14] = 44; h_input[15] = 53;
  checkCudaErrors(cudaMemcpy(d_val_src, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice));

  // predicate call
  predicate_kernel<<<2,8>>>(d_predicate, d_val_src, numElems);

  // copy predicate values to new array
  unsigned int* d_predicate_tmp;
  checkCudaErrors(cudaMalloc((void **) &d_predicate_tmp, ARRAY_BYTES));
  checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

  // set all elements to zero 
  checkCudaErrors(cudaMemset(d_sum_scan, 0, ARRAY_BYTES));

  // sum scan call
  for (int step = 1; step < numElems; step *= 2) {
    exclusive_sum_scan_kernel<<<1, 16>>>(d_sum_scan, d_predicate_tmp, step, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

    printf("round %d\n", step);
    DEBUG(d_sum_scan, ARRAY_BYTES, numElems);
    DEBUG(d_predicate_tmp, ARRAY_BYTES, numElems);
  }

  // debugging
  unsigned int *h_predicate = new unsigned int[numElems];
  unsigned int *h_sum_scan  = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_predicate, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_sum_scan, d_sum_scan,   ARRAY_BYTES, cudaMemcpyDeviceToHost));
 
  printf("INPUT \t PRED \t SCAN\n");
  for (int i = 0; i < numElems; i++)
    printf("%u \t %u \t %u\n", h_input[i], h_predicate[i], h_sum_scan[i]);

  return 0;
}
