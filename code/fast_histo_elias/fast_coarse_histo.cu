
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

// CONSTANTS
const unsigned int NUM_ELEMS    = 1 << 10;
const unsigned int NUM_BINS     = 10;
const unsigned int ARRAY_BYTES  = sizeof(unsigned int) * NUM_ELEMS;

const dim3 BLOCK_SIZE(1024);
const dim3 GRID_SIZE(NUM_BINS);

__global__
void compute_bin_mapping(const unsigned int* const d_in,
                         unsigned int* const d_out) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= NUM_ELEMS) return;

  d_out[mid] = d_in[mid] % NUM_BINS;
}

void init(unsigned int* const h_in) {
  /* initialize random seed: */
  srand(time(NULL));

  /* generate values between 0 and 999: */
  for (int i = 0; i < NUM_ELEMS; i++)
    h_in[i] = rand() % 1000;
}

void print(const unsigned int* const h_in,
           const unsigned int* const h_bin) {
  const unsigned int WIDTH = 4;

  for(int i = 0; i < WIDTH; i++)
    printf("input\tbin\t\t");
  printf("\n");

  for (int i = 0; i < NUM_ELEMS; i++) {
    printf("%u\t%u%s", 
      h_in[i], 
      h_bin[i], 
      ((i % WIDTH == 0) ? "\n" : "\t\t")
    );
}

int main(int argc, char **argv) {
  printf("## STARTING ##");

  // create random values
  unsigned int* h_values = new unsigned int[NUM_ELEMS];
  init(h_values);
  // host memory
  unsigned int* h_bins   = new unsigned int[NUM_ELEMS];
  memset(h_bins, 0, NUM_ELEMS);

  //copy values to device memory
  unsigned int* d_values,
              * d_bins;
  checkCudaErrors(cudaMalloc((void **) &d_values, ARRAY_BYTES));
  checkCudaErrors(cudaMemcpy(d_values, h_values, ARRAY_BYTES, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **) &d_bins, ARRAY_BYTES));

  // compute bin id
  compute_bin_mapping(d_values, d_bins);
  // move memory to host
  checkCudaErrors(cudaMemcpy(h_bins, d_bins, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  // compute coarse bin id

  // sort

  // send coarse bin to each block
  // atomicAdd bins in shared memory
  
  // combine bins and write to global memory


  print(h_values, h_bins);

  printf("## DONE ##");

  return 0;
}
