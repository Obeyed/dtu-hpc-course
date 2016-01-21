
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <string.h>
#include "utils.h"
#include "radix_sort.h"

// CONSTANTS
const unsigned int NUM_ELEMS    = 1 << 6;
const unsigned int NUM_BINS     = 100;
const unsigned int ARRAY_BYTES  = sizeof(unsigned int) * NUM_ELEMS;

const dim3 BLOCK_SIZE(1 << 8);
const dim3 GRID_SIZE(NUM_ELEMS / BLOCK_SIZE.x + 1);

const unsigned int COARSER = NUM_BINS / 10;
const unsigned int MAX_NUMS = 1000;

__global__
void compute_coarse_bin_mapping(const unsigned int* const d_in,
                                unsigned int* const d_out,
                                const size_t COARSE) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= NUM_ELEMS) return;

  d_out[mid] = d_in[mid] / COARSE;
}

__global__
void compute_bin_mapping(const unsigned int* const d_in,
                         unsigned int* const d_out) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= NUM_ELEMS) return;

  d_out[mid] = d_in[mid] % NUM_BINS;
}

void init_rand(unsigned int* const h_in) {
  /* initialize random seed: */
  srand(time(NULL));

  /* generate values between 0 and 999: */
  for (int i = 0; i < NUM_ELEMS; i++)
    h_in[i] = rand() % MAX_NUMS;
}

void print(const unsigned int* const h_in,
           const unsigned int* const h_bins,
           const unsigned int* const h_coarse_bins) {
  const unsigned int WIDTH = 6;

  for(int i = 0; i < WIDTH; i++)
    printf("input\tbin\tcoarse\t\t");
  printf("\n");

  for (int i = 0; i < NUM_ELEMS; i++)
    printf("%u\t%u\t%u%s", 
        h_in[i], 
        h_bins[i], 
        h_coarse_bins[i], 
        ((i % WIDTH == (WIDTH-1)) ? "\n" : "\t\t"));
  printf("\n");
}

void sort(unsigned int* h_coarse_bins, 
          unsigned int* h_bins, 
          unsigned int* h_values) {
  const unsigned int NUM_ARRAYS = 3;
  // set up pointers
  unsigned int** to_be_sorted = new unsigned int*[NUM_ARRAYS];
  to_be_sorted[0] = h_coarse_bins;
  to_be_sorted[1] = h_bins;
  to_be_sorted[2] = h_values;

  unsigned int** sorted = radix_sort(to_be_sorted, NUM_ARRAYS, NUM_ELEMS);

  // update pointers
  h_coarse_bins = sorted[0];
  h_bins = sorted[1];
  h_values = sorted[2];
}

void init_device_memory(unsigned int* const d_values,
                        unsigned int* const d_bins,
                        unsigned int* const d_coarse_bins) {
  checkCudaErrors(cudaMalloc((void **) &d_values, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_bins,   ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_coarse_bins,   ARRAY_BYTES));
}

int main(int argc, char **argv) {
  printf("## STARTING ##\n");
  printf("blocks: %u\tthreads: %u\t coarser: %u", GRID_SIZE.x, BLOCK_SIZE.x, COARSER);

  printf("\n\n");

  // create random values
  unsigned int* h_values = new unsigned int[NUM_ELEMS];
  init_rand(h_values);
  // host memory
  unsigned int* h_bins = new unsigned int[NUM_ELEMS];
  unsigned int* h_coarse_bins = new unsigned int[NUM_ELEMS];

  //copy values to device memory
  unsigned int* d_values, * d_bins, * d_coarse_bins;
  init_device_memory(d_values, d_bins, d_coarse_bins);

  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_values, h_values,  ARRAY_BYTES, cudaMemcpyHostToDevice));

  // compute bin id
  compute_bin_mapping<<<GRID_SIZE, BLOCK_SIZE>>>(d_values, d_bins);
  // move memory to host
  checkCudaErrors(cudaMemcpy(h_bins, d_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  // compute coarse bin id
  compute_coarse_bin_mapping<<<GRID_SIZE, BLOCK_SIZE>>>(d_bins, d_coarse_bins, COARSER);
  // move memory to host
  checkCudaErrors(cudaMemcpy(h_coarse_bins, d_coarse_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  // sort
  printf("BEFORE SORTING:\n");
  print(h_values, h_bins, h_coarse_bins);
  sort(h_coarse_bins, h_bins, h_values);
  printf("AFTER SORTING:\n");
  print(h_values, h_bins, h_coarse_bins);


  // send coarse bin to each block
  // atomicAdd bins in shared memory
  
  // combine bins and write to global memory


  printf("## DONE ##\n");

  return 0;
}

