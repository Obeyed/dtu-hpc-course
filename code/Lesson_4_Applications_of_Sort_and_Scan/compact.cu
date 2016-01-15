// Create predicate array for HW4

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * Calculate if LSB is 0.
 * 1 if true, 0 otherwise.
 */
__global__
void predicate_kernel(unsigned int *d_predicate,
                      unsigned int *d_val_src,
                      const size_t numElems,
                      unsigned int i) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_predicate[mid] = (int)(((d_val_src[mid] & (1 << i)) >> i) == 0);
}

__global__
void inclusive_sum_scan_kernel(unsigned int* d_out,
                               unsigned int* d_in,
                               int step,
                               const size_t numElems) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

	int toAdd = (((mid - step) < 0) ? 0 : d_in[mid - step]);
  d_out[mid] = d_in[mid] + toAdd;
}

__global__
void right_shift_array(unsigned int* d_out,
                       unsigned int* d_in,
                       size_t numElems) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] = (mid == 0) ? 0 : d_in[mid - 1];
}

void DEBUG(unsigned int *device_array, unsigned int ARRAY_BYTES, size_t numElems) {
  unsigned int *h_test  = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_test, device_array, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  for (int i = 0; i < numElems; i++)
    printf("%u ", h_test[i]);
  printf("\n");
}

__global__ 
void reduce_kernel(unsigned int * d_out, unsigned int * d_in, int size) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  for (unsigned int s = blockDim.x / 2; s>0; s>>=1) {
    if ((tid < s) && (pos+s < size))
      d_in[pos] = d_in[pos] + d_in[pos+s];
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid == 0) && (pos < size))
    d_out[blockIdx.x] = d_in[pos];
}

void reduce_wrapper(unsigned int * d_out, unsigned int * d_in, int size, int num_threads) {
  int num_blocks = size / num_threads + 1;

  unsigned int * d_tmp;
  checkCudaErrors(cudaMalloc(&d_tmp, sizeof(int)*num_blocks));
  checkCudaErrors(cudaMemset(d_tmp, 0, sizeof(int)*num_blocks));

  int prev_num_blocks;
  int remainder = 0;
  // recursively solving, will run approximately log base num_threads times.
  do {
    reduce_kernel<<<num_blocks, num_threads>>>(d_tmp, d_in, size);

    remainder = size % num_threads;
    size = size / num_threads + remainder;

    // updating input to intermediate
    checkCudaErrors(cudaMemcpy(d_in, d_tmp, sizeof(int)*num_blocks, cudaMemcpyDeviceToDevice));

    // Updating num_blocks to reflect how many blocks we now want to compute on
    prev_num_blocks = num_blocks;
    num_blocks = size / num_threads + 1;      

    // updating intermediate
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaMalloc(&d_tmp, sizeof(int)*num_blocks));
  } while(size > num_threads);

  // computing rest
  reduce_kernel<<<1, size>>>(d_out, d_in, prev_num_blocks);
}

__global__
void map_kernel(unsigned int* d_out,
                unsigned int* d_in,
                unsigned int* d_predicate,
                unsigned int* d_sum_scan_0,
                unsigned int* d_sum_scan_1,
                size_t numElems) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  int pos;

  if (d_predicate[mid])
    pos = d_sum_scan_0[mid];
  else 
    pos = d_sum_scan_1[mid];

  d_out[pos] = d_in[mid];
}


void exclusive_sum_scan(unsigned int* d_out,
                        unsigned int* d_predicate,
                        unsigned int* d_predicate_tmp,
                        unsigned int* d_sum_scan,
                        unsigned int ARRAY_BYTES,
                        size_t numElems,
                        int GRID_SIZE,
                        int BLOCK_SIZE) {
  // copy predicate values to new array
  checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  // set all elements to zero 
  checkCudaErrors(cudaMemset(d_sum_scan, 0, ARRAY_BYTES));

  // sum scan call
  for (int step = 1; step < numElems; step *= 2) {
    inclusive_sum_scan_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_sum_scan, d_predicate_tmp, step, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  }

  // shift to get exclusive scan
  checkCudaErrors(cudaMemcpy(d_out, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  right_shift_array<<<GRID_SIZE,BLOCK_SIZE>>>(d_out, d_sum_scan, numElems);
}

__global__
void toggle_predicate_kernel(unsigned int* const d_out, 
                             const unsigned int* const d_predicate,
                             const size_t numElems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] = ((d_predicate[mid]) ? 0 : 1);
}

__global__
void add_splitter_map_kernel(unsigned int* const d_out,
                             const unsigned int* const shift, 
                             const size_t numElems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] += shift[0];
}

int main(void) {
  const size_t numElems = 1 << 10;
  const int BLOCK_SIZE  = 512;
  const int GRID_SIZE   = numElems / BLOCK_SIZE + 1;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * numElems;
  const unsigned int BITS_PER_BYTE = 8;

  // host memory
  unsigned int* const h_input  = new unsigned int[numElems];
  unsigned int* const h_output = new unsigned int[numElems];

  srand(time(NULL));
  for (unsigned int i = 0; i < numElems; i++)
    h_input[i] = rand(); 

  // device memory
  unsigned int *d_val_src, *d_predicate, *d_sum_scan, *d_predicate_tmp, *d_sum_scan_0, *d_sum_scan_1, *d_predicate_toggle, *d_reduce, *d_map;
  checkCudaErrors(cudaMalloc((void **) &d_val_src,          ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_map,              ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate,        ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate_tmp,    ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate_toggle, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan,         ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan_0,       ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan_1,       ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_reduce, sizeof(unsigned int)));

  // copy host array to device
  checkCudaErrors(cudaMemcpy(d_val_src, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice));

  for (unsigned int i = 0; i < (BITS_PER_BYTE * sizeof(unsigned int)); i++) {
    // predicate is that LSB is 0
    predicate_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_predicate, d_val_src, numElems, i);

    // calculate scatter addresses from predicates
    exclusive_sum_scan(d_sum_scan_0, d_predicate, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, numElems, GRID_SIZE, BLOCK_SIZE);

    // copy contents of predicate, so we do not change its content
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

    // calculate how many elements had predicate equal to 1
    reduce_wrapper(d_reduce, d_predicate_tmp, numElems, BLOCK_SIZE);

    // toggle predicate values, so we can compute scatter addresses for toggled predicates
    toggle_predicate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_predicate_toggle, d_predicate, numElems);
    // so we now have addresses for elements where LSB is equal to 1
    exclusive_sum_scan(d_sum_scan_1, d_predicate_toggle, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, numElems, GRID_SIZE, BLOCK_SIZE);
    // shift scatter addresses according to amount of elements that had LSB equal to 0
    add_splitter_map_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_sum_scan_1, d_reduce, numElems);

    // move elements accordingly
    map_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_map, d_val_src, d_predicate, d_sum_scan_0, d_sum_scan_1, numElems);

    // swap pointers, instead of moving elements
    std::swap(d_val_src, d_map);
  }

  // debugging
  checkCudaErrors(cudaMemcpy(h_output, d_val_src, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  for (int i = 0; i < numElems; i++)
    printf("%u%s", h_output[i], ((i % 8 == 7) ? "\n" : "\t"));

  return 0;
}
