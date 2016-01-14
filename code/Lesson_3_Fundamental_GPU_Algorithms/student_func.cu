/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

unsigned int set_grid(unsigned int SIZE, unsigned int BLOCK_SIZE)
{
  return SIZE/BLOCK_SIZE + 1;//((SIZE % BLOCK_SIZE)? 1 : 0);
}

__global__ void excluding_kernel(unsigned int * d_out, unsigned int * d_in, unsigned int size)
{
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if(mid >= size) return;
  if(mid == 0){d_out[mid] = 0; return;}
  d_out[mid] = d_in[mid]-1;
}

__global__ void simple_histo(unsigned int *d_bins, const float * const d_in, const int BIN_COUNT, int ARRAY_SIZE, float min_logLum, float range_logLum)
{
  unsigned int mid = threadIdx.x + blockDim.x + blockIdx.x;
  // checking for out-of-bounds
  if (mid>=ARRAY_SIZE) return;

  unsigned int myBin = min(static_cast<unsigned int>(BIN_COUNT - 1),
                           static_cast<unsigned int>(((d_in[mid]-min_logLum) / range_logLum) * BIN_COUNT));
  atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void reduce_minmax_kernel(float * d_out, float * d_in, unsigned int SIZE, int choice)
{
  unsigned int mid = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;
  for (unsigned int s = blockDim.x / 2; s>0; s>>=1)
  {
    if ((tid < s) && (mid+s < SIZE))
    {
      if(choice)
        d_in[mid] = max(d_in[mid], d_in[mid+s]);
      else
        d_in[mid] = min(d_in[mid], d_in[mid+s]);
    }
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid==0) && (mid < SIZE))
    d_out[blockIdx.x] = d_in[mid];
}

float reduce_minmax(const float * const d_logLuminance, unsigned int SIZE, unsigned int BYTES,unsigned int BLOCK_SIZE, int choice)
{
  unsigned int GRID_SIZE = set_grid(SIZE, BLOCK_SIZE);
  float * d_intermediate_in;
  float * d_intermediate_out;
  cudaMalloc(&d_intermediate_in, BYTES);
  cudaMalloc(&d_intermediate_out, BYTES);
  cudaMemcpy(d_intermediate_in, d_logLuminance, BYTES, cudaMemcpyDeviceToDevice);

  do{
     reduce_minmax_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_intermediate_out, d_intermediate_in, SIZE, choice);
     SIZE = GRID_SIZE;
     cudaMemcpy(d_intermediate_in, d_intermediate_out, SIZE*sizeof(float), cudaMemcpyDeviceToDevice);
     GRID_SIZE = set_grid(SIZE, BLOCK_SIZE);
  }while(SIZE>BLOCK_SIZE);

  reduce_minmax_kernel<<<1, SIZE>>>(d_intermediate_out, d_intermediate_in, SIZE, choice);
  float minmax;
  cudaMemcpy(&minmax, d_intermediate_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_intermediate_in);
  cudaFree(d_intermediate_out);
  return minmax;
}

__global__ void hs_kernel_global(unsigned int * d_out, unsigned int * d_in, int step, unsigned int SIZE)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if(myId >= SIZE) return;
  d_out[myId] = d_in[myId] + (((myId - step)<0) ? 0 : d_in[myId-step]);
}

void hs_kernel_wrapper(unsigned int * const d_out, unsigned int * d_in, unsigned int SIZE, unsigned int BYTES, unsigned int BLOCK_SIZE)
{
  unsigned int GRID_SIZE = set_grid(SIZE, BLOCK_SIZE);
//  unsigned int * d_intermediate;
//  cudaMalloc((void **) &d_intermediate, BYTES);
//  cudaMemcpy(d_intermediate, d_in, BYTES, cudaMemcpyDeviceToDevice);

  for(int step = 1; step < SIZE; step<<=1)
  {
    hs_kernel_global<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, step, SIZE);
    checkCudaErrors(cudaMemcpy(d_in, d_out, BYTES, cudaMemcpyDeviceToDevice));
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  unsigned int ARRAY_SIZE = numRows * numCols,
               ARRAY_BYTES = sizeof(float) * ARRAY_SIZE,
               BIN_BYTES = sizeof(float) * numBins,
               BLOCK_SIZE = 1024,
               GRID_SIZE_ARRAY = set_grid(ARRAY_SIZE, BLOCK_SIZE),
               GRID_SIZE_BINS = set_grid(numBins, BLOCK_SIZE);

  unsigned int * d_bins, *d_bins_excluding;

  // 1)
  min_logLum = reduce_minmax(d_logLuminance, ARRAY_SIZE, ARRAY_BYTES, BLOCK_SIZE, 0);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  max_logLum = reduce_minmax(d_logLuminance, ARRAY_SIZE, ARRAY_BYTES, BLOCK_SIZE, 1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 2)
  const float range_logLum = max_logLum - min_logLum;

  printf("got min of %.5f\n", min_logLum);
  printf("got max of %.5f\n", max_logLum);
  printf("got range of %.5f\n", range_logLum);
  printf("numBins %d\n", numBins);

  // 3)
  checkCudaErrors(cudaMalloc(&d_bins, BIN_BYTES));    
  checkCudaErrors(cudaMemset(d_bins, 0, BIN_BYTES));
  simple_histo<<<GRID_SIZE_ARRAY, BLOCK_SIZE>>>(d_bins, d_logLuminance, numBins, ARRAY_SIZE, min_logLum, range_logLum);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  unsigned int h_out[numBins];
  cudaMemcpy(&h_out, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 100; i++)
  {
      printf("hist out %d\n", h_out[i]);
  }
  // Using H&S, so making it "excluding"
  cudaMalloc(&d_bins_excluding, BIN_BYTES);
  excluding_kernel<<<GRID_SIZE_BINS, BLOCK_SIZE>>>(d_bins_excluding, d_bins, numBins);
  // 4)
  hs_kernel_wrapper(d_cdf, d_bins, numBins, BIN_BYTES, BLOCK_SIZE);
  cudaMemcpy(h_out, d_cdf, BIN_BYTES, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 100; i++)
  {
      printf("hist out %d\n", h_out[i]);
  }
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
}