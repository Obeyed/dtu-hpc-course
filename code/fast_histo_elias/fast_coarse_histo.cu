
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

const dim3 BLOCK_SIZE(1 << 8);
const dim3 GRID_SIZE(NUM_ELEMS / BLOCK_SIZE.x + 1);

const unsigned int COARSER_SIZE = NUM_BINS / 10;
const unsigned int COARSER_BYTES = sizeof(unsigned int) * COARSER_SIZE;
const unsigned int MAX_NUMS = 1000;

__global__
void fire_up_local_bins(unsigned int* const d_out,
                        const unsigned int* const d_bins,
                        const unsigned int l_start,
                        const int l_end) {
  if (l_end < 0) return; // means that no values are in coarsed bin

  const unsigned int l_pos = l_start + threadIdx.x + blockIdx.x * blockDim.x;
  if (l_pos < l_start || l_pos >= l_end) return;

  const unsigned int bin = d_bins[l_pos];
  // read some into shared memory
  // atomic adds
  // write to global memory
  atomicAdd(&(d_out[bin]), 1);
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
           const unsigned int* const h_coarse_bins,
           const unsigned int* const h_positions) {
  const unsigned int WIDTH = 4;

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

  printf("positions:\n");
  for (int i = 0; i < COARSER_SIZE; i++)
    printf("%u\t", h_positions[i]);
  printf("\n");
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

int main(int argc, char **argv) {
  printf("## STARTING ##\n");
  printf("blocks: %u\tthreads: %u\t COARSER_SIZE: %u", GRID_SIZE.x, BLOCK_SIZE.x, COARSER_SIZE);
  printf("\n\n");

  // create random values
  unsigned int* h_values = new unsigned int[NUM_ELEMS];
  init_rand(h_values);
  // host memory
  unsigned int* h_bins = new unsigned int[NUM_ELEMS];
  unsigned int* h_coarse_bins = new unsigned int[NUM_ELEMS];
  unsigned int* h_histogram = new unsigned int[NUM_BINS];
  unsigned int* h_positions = new unsigned int[COARSER_SIZE];

  //copy values to device memory
  unsigned int* d_values, * d_bins, * d_coarse_bins, * d_positions;
  checkCudaErrors(cudaMalloc((void **) &d_values, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_bins,   ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_coarse_bins, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_positions, ARRAY_BYTES));

  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_values, h_values,  ARRAY_BYTES, cudaMemcpyHostToDevice));

  // compute bin id
  compute_bin_mapping<<<GRID_SIZE, BLOCK_SIZE>>>(d_values, d_bins);
  // move memory to host
  checkCudaErrors(cudaMemcpy(h_bins, d_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  // compute coarse bin id
  compute_coarse_bin_mapping<<<GRID_SIZE, BLOCK_SIZE>>>(d_bins, d_coarse_bins, COARSER_SIZE);
  // move memory to host
  checkCudaErrors(cudaMemcpy(h_coarse_bins, d_coarse_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  // sort
  sort(h_coarse_bins, h_bins, h_values);
  checkCudaErrors(cudaMemcpy(d_coarse_bins, h_coarse_bins,  ARRAY_BYTES, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_bins,        h_bins,         ARRAY_BYTES, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_values,      h_values,       ARRAY_BYTES, cudaMemcpyHostToDevice));

  // find starting position for each coarsed bin
  cudaMemset(d_positions, 0, COARSER_BYTES);
  checkCudaErrors(cudaMemcpy(h_positions, d_positions, COARSER_BYTES, cudaMemcpyDeviceToHost));

  // find positions of separators
  find_positions_mapping_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_positions, d_coarse_bins);
  checkCudaErrors(cudaMemcpy(h_positions, d_positions, COARSER_BYTES, cudaMemcpyDeviceToHost));
  
  print(h_values, h_bins, h_coarse_bins, h_positions);

  // ####
  unsigned int* d_bin_grid;
  // created entire bin grid in first run
  // only access relevant elements in kernel
  // based on bin_size and bin_start
  checkCudaErrors(cudaMalloc((void **) &d_bin_grid, TOTAL_BIN_BYTES));
  checkCudaErrors(cudaMemset(d_bin_grid, 0, TOTAL_BIN_BYTES));

  // make some local bins
  unsigned int local_bin_start = 0;
  unsigned int local_bin_end = h_positions[1];
  int amount = local_bin_end - local_bin_start;
  // calculate local grid size
  unsigned int grid_size = local_bin_end / BLOCK_SIZE.x + 1;

  fire_up_local_bins<<<grid_size, BLOCK_SIZE>>>(d_bin_grid, d_bins, local_bin_start, local_bin_end);

  for (unsigned int i = 1; i < COARSER_SIZE - 1; i++) {
    // make some local bins
    local_bin_start = h_positions[i];
    local_bin_end   = h_positions[i+1];
    amount = local_bin_end - local_bin_start;
    // calculate local grid size
    grid_size = local_bin_end / BLOCK_SIZE.x + 1;

    if (amount > 0)
      fire_up_local_bins<<<grid_size, BLOCK_SIZE>>>(d_bin_grid, d_bins, local_bin_start, local_bin_end);
  }


  // do final loop
  local_bin_start = h_positions[COARSER_SIZE-1];
  local_bin_end   = NUM_ELEMS;
  amount = local_bin_end - local_bin_start;
  // calculate local grid size
  grid_size = local_bin_end / BLOCK_SIZE.x + 1;

  if (amount > 0)
    fire_up_local_bins<<<grid_size, BLOCK_SIZE>>>(d_bin_grid, d_bins, local_bin_start, local_bin_end);

  checkCudaErrors(cudaMemcpy(h_histogram, d_bin_grid, TOTAL_BIN_BYTES, cudaMemcpyDeviceToHost));

  printf("\n");
  for (int j = 0; j < NUM_BINS; j++)
    printf("%d:%u\t%s", 
        j,
        h_histogram[j], 
        ((j % 6 == 5) ? "\n" : "\t\t"));
  printf("\n");

  //#####

  cudaFree(d_bin_grid); cudaFree(d_values); cudaFree(d_positions);
  cudaFree(d_coarse_bins); cudaFree(d_bins);

  printf("## DONE ##\n");

  return 0;
}

