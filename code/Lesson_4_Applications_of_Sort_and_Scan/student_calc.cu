//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void histo_kernel(unsigned int *d_binHisto, 
                  unsigned int * const d_valsSrc, 
                  int mask, 
                  const size_t numElems, 
                  int i) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;

  if (mid >= numElems) return;

  unsigned int bin = (d_valsSrc[mid] & mask) >> i;
  atomicAdd(&(d_binHisto[bin]), 1);
}

__global__
void scan_kernel(unsigned int *d_binHisto,
                 unsigned int *d_binScan,
                 unsigned int step,
                 const int numBins) {
  unsigned int tid = threadIdx.x;

  if ((tid == 0) || (tid > (numBins - 1)))
    return;

  d_binScan[tid] = d_binScan[tid - 1] + d_binHisto[tid - 1];
}

__global__
void map_kernel(unsigned int * const d_valsDst, 
                unsigned int * const d_posDst,
                unsigned int * const d_valsSrc, 
                unsigned int * const d_posSrc,
                unsigned int *d_binScan,
                unsigned int mask,
                const size_t numElems,
                unsigned int i,
                const int numBins) {
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;

  if (mid >= numElems) return;

  unsigned int bin = (d_valsSrc[mid] & mask) >> i;
  unsigned int pos = atomicInc(&(d_binScan[bin]), 1);

  d_valsDst[pos] = d_valsSrc[mid];
  d_posDst[pos]  = d_posSrc[mid];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  const int numBits = 1;
  const int numBins = 1 << numBits;
  const int BITS_PER_BYTE = 8;
  const int BIN_BYTES = sizeof(unsigned int) * numBins;
  const int ARRAY_BYTES = sizeof(unsigned int) * numElems;
  const int BLOCK_SIZE = 512;
  const int GRID_SIZE  = (numElems / BLOCK_SIZE) + 1;

  // initialise device memory
  unsigned int *d_binHisto, *d_binScan;
  checkCudaErrors(cudaMalloc((void **) &d_binHisto, BIN_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_binScan,  BIN_BYTES));

  unsigned int * d_valsSrc = d_inputVals; 
  unsigned int * d_valsDst = d_outputVals;
  unsigned int * d_posSrc  = d_inputPos;
  unsigned int * d_posDst  = d_outputPos;

  // loop through each bit
  for (unsigned int i = 0; i < BITS_PER_BYTE * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;
    // reset all memory locations
    checkCudaErrors(cudaMemset(d_binHisto, 0, BIN_BYTES));
    checkCudaErrors(cudaMemset(d_binScan,  0, BIN_BYTES));

    // build histo
    histo_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_binHisto, d_valsSrc, mask, numElems, i);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // build scan
    for (int step = 1; step < numBins; step <<= 1) {
      scan_kernel<<<1, numBins>>>(d_binHisto, d_binScan, step, numBins);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaMemcpy(d_binHisto, d_binScan, BIN_BYTES, cudaMemcpyDeviceToDevice));
    }

    map_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_valsDst, d_posDst, d_valsSrc, d_posSrc, d_binScan, mask, numElems, i, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // swap pointers
    std::swap(d_valsSrc, d_valsDst);
    std::swap(d_posSrc, d_posDst);
  }
  // copy values to output (not sure why this is necessary)
  cudaMemcpy(d_outputVals, d_inputVals, BIN_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_outputPos, d_inputPos, BIN_BYTES, cudaMemcpyDeviceToDevice);
             
  checkCudaErrors(cudaFree(d_binScan)); 
  checkCudaErrors(cudaFree(d_binHisto));
}

int main(void) {
  size_t numElems = 10;
  unsigned int* d_inputVals;
  unsigned int* d_inputPos;
  unsigned int* d_outputVals;
  unsigned int* d_outputPos;
  const int ARRAY_BYTES = sizeof(unsigned int) * numElems;

  checkCudaErrors(cudaMalloc((void **) &d_inputVals,  ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_outputVals, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_inputPos,   ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_outputPos,  ARRAY_BYTES));

  unsigned int *h_input = new unsigned int[numElems];
  unsigned int *h_pos = new unsigned int[numElems];

  h_input[0] = 10;
  h_input[1] = 5;
  h_input[2] = 3;
  h_input[3] = 4;
  h_input[4] = 6;
  h_input[5] = 1;
  h_input[6] = 2;
  h_input[7] = 9;
  h_input[8] = 8;
  h_input[9] = 7;

  for (int i = 0; i < numElems; i++)
    h_pos[i] = i;

  checkCudaErrors(cudaMemcpy(d_inputVals, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_inputPos,  h_pos,   ARRAY_BYTES, cudaMemcpyHostToDevice));

  your_sort(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
  
  // debugging
  unsigned int *h_result = new unsigned int[numElems];
  unsigned int *h_outPos = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_result, d_outputVals, ARRAY_BYTES, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_outPos, d_outputPos,  ARRAY_BYTES, cudaMemcpyDeviceToHost));
    
  printf("RESULTS: \n");
  for (int i = 0; i < numElems; i++)
    printf("%u", h_result[i]);

  return 0;
}
