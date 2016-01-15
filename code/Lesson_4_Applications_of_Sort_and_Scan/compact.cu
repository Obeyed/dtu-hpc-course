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
void toggle_predicate_kernel(unsigned int* d_out, 
                             unsigned int* d_predicate,
                             size_t numElems) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] = ((d_predicate[mid]) ? 0 : 1);
}

__global__
void add_splitter_map_kernel(unsigned int* d_out,
                             unsigned int* shift, 
                             size_t numElems) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] += shift[0];
}

int main(void) {
  size_t numElems = 16;
  int GRID_SIZE = 2;
  int BLOCK_SIZE = 8;
  unsigned int ARRAY_BYTES = sizeof(unsigned int) * numElems;

  // device memory
  unsigned int *d_val_src, *d_predicate, *d_sum_scan;
  checkCudaErrors(cudaMalloc((void **) &d_val_src,   ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan,  ARRAY_BYTES));
  unsigned int* d_predicate_tmp;
  checkCudaErrors(cudaMalloc((void **) &d_predicate_tmp, ARRAY_BYTES));

  // input array
  unsigned int* h_input = new unsigned int[numElems];
  h_input[0]  = 39; h_input[1]  = 21; h_input[2]  = 84; h_input[3]  = 40;
  h_input[4]  = 78; h_input[5]  = 85; h_input[6]  = 13; h_input[7]  = 4;
  h_input[8]  = 47; h_input[9]  = 45; h_input[10] = 91; h_input[11] = 60;
  h_input[12] = 74; h_input[13] = 8;  h_input[14] = 44; h_input[15] = 53;
  checkCudaErrors(cudaMemcpy(d_val_src, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice));

  const unsigned int BITS_PER_BYTE = 8;

  // LOOP START
  for (unsigned int i = 0; BITS_PER_BYTE * sizeof(unsigned int); i++) {
    //##########
    // predicate call
    predicate_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_predicate, d_val_src, numElems, i);
    //##########

    //##########
    // LSB == 0
    unsigned int* d_sum_scan_0;
    checkCudaErrors(cudaMalloc((void **) &d_sum_scan_0, ARRAY_BYTES));
    exclusive_sum_scan(d_sum_scan_0, d_predicate, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, numElems, GRID_SIZE, BLOCK_SIZE);
    //##########

    //##########
    // reduce to get amount of LSB equal to 0
    unsigned int* d_reduce;
    checkCudaErrors(cudaMalloc((void **) &d_reduce, sizeof(unsigned int)));
    // copy contents 
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
    reduce_wrapper(d_reduce, d_predicate_tmp, numElems, BLOCK_SIZE);
    unsigned int h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_reduce, sizeof(int), cudaMemcpyDeviceToHost));
    //##########

    //##########
    // LSB == 1
    unsigned int* d_sum_scan_1;
    unsigned int* d_predicate_toggle;
    checkCudaErrors(cudaMalloc((void **) &d_sum_scan_1,       ARRAY_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_predicate_toggle, ARRAY_BYTES));
    // flip predicate values
    toggle_predicate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_predicate_toggle, d_predicate, numElems);
    exclusive_sum_scan(d_sum_scan_1, d_predicate_toggle, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, numElems, GRID_SIZE, BLOCK_SIZE);
    // map sum_scan_1 to add splitter
    add_splitter_map_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_sum_scan_1, d_reduce, numElems);
    //##########

    //##########
    // move elements accordingly
    unsigned int* d_map;
    checkCudaErrors(cudaMalloc((void **) &d_map, ARRAY_BYTES));
    map_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_map, d_val_src, d_predicate, d_sum_scan_0, d_sum_scan_1, numElems);
    //##########

    // debugging
    unsigned int *h_predicate   = new unsigned int[numElems];
    unsigned int *h_predicate_toggle = new unsigned int[numElems];
    unsigned int *h_sum_scan_0  = new unsigned int[numElems];
    unsigned int *h_sum_scan_1  = new unsigned int[numElems];
    unsigned int *h_map         = new unsigned int[numElems];
    checkCudaErrors(cudaMemcpy(h_predicate,   d_predicate,  ARRAY_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_predicate_toggle, d_predicate_toggle,  ARRAY_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sum_scan_0,  d_sum_scan_0, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sum_scan_1,  d_sum_scan_1, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_map,         d_map,        ARRAY_BYTES, cudaMemcpyDeviceToHost));
   
    printf("INPUT\tPRED\tPRED_T\tSCAN_0\tSCAN_1\tMAP\n");
    for (int i = 0; i < numElems; i++)
      printf("%u\t%u\t%u\t%u\t%u\t%u\n", h_input[i], h_predicate[i], h_predicate_toggle[i], h_sum_scan_0[i], h_sum_scan_1[i], h_map[i]);

    printf("sum: \t %u\n", h_result);
  }
  // LOOP END

  return 0;
}
