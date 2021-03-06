
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <string.h>
#include "utils.h"
#include "radix_sort.h"

// CONSTANTS
const unsigned int NUM_ELEMS    = 1 << 10;
const unsigned int NUM_BINS     = 100;
const unsigned int ARRAY_BYTES  = sizeof(unsigned int) * NUM_ELEMS;
const unsigned int TOTAL_BIN_BYTES  = sizeof(unsigned int) * NUM_BINS;

const dim3 BLOCK_SIZE(1 << 7);
const dim3 GRID_SIZE(NUM_ELEMS / BLOCK_SIZE.x + 1);

const unsigned int COARSE_SPACE = 10;
const unsigned int COARSE_SIZE = NUM_BINS / COARSE_SPACE;
const unsigned int COARSER_BYTES = sizeof(unsigned int) * COARSE_SIZE;
const unsigned int MAX_NUMS = 1000;


__global__
void parallel_reference_calc(unsigned int* const d_out,
                             const unsigned int* const d_in) {
  for (unsigned int l = 0; l < NUM_ELEMS; ++l)
    d_out[(d_in[l] % NUM_BINS)]++;
}


__global__
void coarse_histogram_count(unsigned int* const d_out,
                        const unsigned int* const d_bins,
                        const unsigned int l_start,
                        const unsigned int l_end,
                        const unsigned int COARSE_SIZE,
                        const unsigned int COARSE_SPACE,
                        const unsigned int num_coarse) {

  const unsigned int l_pos = l_start + threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int start_place = blockIdx.x*COARSE_SPACE;
  if (l_pos < l_start || l_pos >= l_end) return;

  const unsigned int bin = d_bins[l_pos] - COARSE_SIZE*num_coarse;
  // read some into shared memory
  // atomic adds
  // write to global memory
  atomicAdd(&(d_out[start_place + bin]), 1);
}

__global__
void compute_coarse_bin_mapping(const unsigned int* const d_in,
                                unsigned int* const d_out,
                                const size_t COARSE) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= NUM_ELEMS) return;

  d_out[mid] = d_in[mid] / COARSE;
}

__global__
void compute_bin_mapping(const unsigned int* const d_in,
                         unsigned int* const d_out) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= NUM_ELEMS) return;

  d_out[mid] = d_in[mid] % NUM_BINS;
}

__global__
void find_positions_mapping_kernel(unsigned int* const d_out,
                                   const unsigned int* const d_in) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if ((mid >= NUM_ELEMS) || (mid == 0)) return;

  if (d_in[mid] != d_in[mid-1])
    d_out[d_in[mid]] = mid;
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
           const unsigned int* const h_histogram) {
  const unsigned int WIDTH = 5;

  for(int i = 0; i < WIDTH; i++)
    printf("input\tbin\t\t");
  printf("\n");

  for (int i = 0; i < NUM_ELEMS; i++)
    printf("%u\t%u%s", 
        h_in[i], 
        h_bins[i], 
        ((i % WIDTH == (WIDTH-1)) ? "\n" : "\t\t"));
  printf("\n");

  printf("histogram:\n");
  for (int i = 0; i < NUM_BINS; i++)
    printf("%u%s", 
        h_histogram[i], 
        ((i % 15 == 14) ? "\n" : "\t"));
  printf("\n\n");
}

void sort(unsigned int*& h_coarse_bins, 
          unsigned int*& h_bins, 
          unsigned int*& h_values) {
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

void init_memory(unsigned int*& h_values,
                 unsigned int*& h_bins,
                 unsigned int*& h_coarse_bins,
                 unsigned int*& h_histogram,
                 unsigned int*& h_positions,
                 unsigned int*& h_reference_histo,
                 unsigned int*& d_values,
                 unsigned int*& d_bins,
                 unsigned int*& d_coarse_bins,
                 unsigned int*& d_positions,
                 unsigned int*& d_bin_grid,
                 unsigned int*& d_histogram) {
  // host
  h_values          = new unsigned int[NUM_ELEMS];
  h_bins            = new unsigned int[NUM_ELEMS];
  h_coarse_bins     = new unsigned int[NUM_ELEMS];
  h_histogram       = new unsigned int[NUM_BINS];
  h_reference_histo = new unsigned int[NUM_BINS];
  h_positions       = new unsigned int[COARSE_SIZE];
  // device
  checkCudaErrors(cudaMalloc((void **) &d_values,       ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_bins,         ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_coarse_bins,  ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_positions,    ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_bin_grid,     TOTAL_BIN_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_histogram,    TOTAL_BIN_BYTES));
}

void streamed_coarse_atomic_bin_calc(unsigned int*& d_values,
                            unsigned int*& h_values,
                            unsigned int*& d_bins,
                            unsigned int*& h_bins,
                            unsigned int*& d_coarse_bins,
                            unsigned int*& h_coarse_bins,
                            unsigned int*& d_positions,
                            unsigned int*& h_positions,
                            unsigned int*& d_bin_grid,
                            unsigned int*& h_histogram) {
  // compute bin id
  compute_bin_mapping<<<GRID_SIZE, BLOCK_SIZE>>>(d_values, d_bins);

  // compute coarse bin id
  compute_coarse_bin_mapping<<<GRID_SIZE, BLOCK_SIZE>>>(d_bins, d_coarse_bins, COARSE_SIZE);
  // move memory to host
  checkCudaErrors(cudaMemcpy(h_coarse_bins, d_coarse_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  // move memory to host
  checkCudaErrors(cudaMemcpy(h_bins, d_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));
  // sort
  sort(h_coarse_bins, h_bins, h_values);
  checkCudaErrors(cudaMemcpy(d_coarse_bins, h_coarse_bins,  ARRAY_BYTES, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_bins,        h_bins,         ARRAY_BYTES, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_values,      h_values,       ARRAY_BYTES, cudaMemcpyHostToDevice));

  // find starting position for each coarsed bin
  checkCudaErrors(cudaMemset(d_positions, 0, COARSER_BYTES));
  checkCudaErrors(cudaMemcpy(h_positions, d_positions, COARSER_BYTES, cudaMemcpyDeviceToHost));

  // find positions of separators
  find_positions_mapping_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_positions, d_coarse_bins);
  checkCudaErrors(cudaMemcpy(h_positions, d_positions, COARSER_BYTES, cudaMemcpyDeviceToHost));
  
  for (unsigned int i = 0; i < 1; i++) {//COARSE_SIZE; i++) {
    // make some local bins
    const unsigned int local_bin_start = h_positions[i];
    const unsigned int local_bin_end   = h_positions[i+1];
    const unsigned int amount = local_bin_end - local_bin_start;
    // calculate local grid size
    unsigned int grid_size = amount / BLOCK_SIZE.x + 1;
    checkCudaErrors(cudaMemset(d_bin_grid, 0, TOTAL_BIN_BYTES*grid_size));
    if (amount > 0) {
      // call kernel with stream
      coarse_histogram_count<<<grid_size, BLOCK_SIZE>>>(d_bin_grid, d_bins, local_bin_start, local_bin_end, COARSE_SIZE, COARSE_SPACE, i);
    }
  }
  // make sure device is cleared
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(h_histogram, d_bin_grid, TOTAL_BIN_BYTES, cudaMemcpyDeviceToHost));
  for(int j=0; j<NUM_BINS; j++) printf("%d ", h_histogram[j]);
}


int main(int argc, char **argv) {
  // host pointers
  unsigned int* h_values, * h_bins, * h_coarse_bins, * h_histogram, * h_positions, * h_reference_histo;
  // device pointers
  unsigned int* d_values, * d_bins, * d_coarse_bins, * d_positions, * d_bin_grid, * d_histogram;
  // set up memory
  init_memory(h_values, h_bins, h_coarse_bins, h_histogram, h_positions, h_reference_histo,
              d_values, d_bins, d_coarse_bins, d_positions, d_bin_grid, d_histogram);

  // initialise random values
  init_rand(h_values);
//  memset(h_bins, 0, TOTAL_BIN_BYTES);
  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_values, h_values,  ARRAY_BYTES, cudaMemcpyHostToDevice));

  //###
  // reset output array
  checkCudaErrors(cudaMemset(d_histogram, 0, TOTAL_BIN_BYTES));
  // parallel reference test
//  parallel_reference_calc<<<1,1>>>(d_histogram, d_values);
//  checkCudaErrors(cudaMemcpy(h_reference_histo, d_histogram,  TOTAL_BIN_BYTES, cudaMemcpyDeviceToHost));
  //###

/*  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, TOTAL_BIN_BYTES, cudaMemcpyDeviceToHost));

  //###
  coarse_atomic_bin_calc(d_values, h_values, d_bins, h_bins, d_coarse_bins, h_coarse_bins, 
                         d_positions, h_positions, d_bin_grid, h_histogram);
  printf("COARSE ATOMIC BIN (%s)\n", 
      (compare_results(h_reference_histo, h_histogram) ? "Success" : "Failed"));
  //###
*/

  //###
  streamed_coarse_atomic_bin_calc(d_values, h_values, d_bins, h_bins, d_coarse_bins, h_coarse_bins, 
                                  d_positions, h_positions, d_bin_grid, h_histogram);
//  printf("STREAMED COARSE ATOMIC BIN (%s)\n", 
//      (compare_results(h_reference_histo, h_histogram) ? "Success" : "Failed"));
  //###

  cudaFree(d_bin_grid); cudaFree(d_values); cudaFree(d_positions);
  cudaFree(d_coarse_bins); cudaFree(d_bins); cudaFree(d_histogram);
  cudaDeviceReset();

  return 0;
}
