#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>


// Computes the sum of elements in d_in in shared memory
__global__ 
void shared_reduce_kernel(unsigned int* const d_out,
                          unsigned int* const d_in,
                          const size_t NUM_ELEMS) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  if (pos >= NUM_ELEMS) return;

  extern __shared__ unsigned int sdata[]; // allocate shared memory
  sdata[tid] = d_in[pos];                 // each thread loads global to shared
  __syncthreads();                        // make sure all threads are done

  for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
    if ((tid < s) && ((pos + s) < NUM_ELEMS))
      sdata[tid] +=  sdata[tid + s];      // perform operations on shared memory
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid == 0) && (pos < NUM_ELEMS))
    d_out[blockIdx.x] = sdata[0];         // copy shared back to global
}

/* -------- KERNEL -------- */
__global__ void reduce_kernel(unsigned int * d_out, unsigned int * d_in, unsigned int SIZE)
{
  // position and threadId
  unsigned int tid = threadIdx.x;
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;

  // do reduction in global memory
  for (unsigned int s = blockDim.x / 2; s>0; s>>=1)
  {
    if ((tid < s) && (mid+s < SIZE)) // Handling out of bounds
        d_in[mid] = d_in[mid] + d_in[mid+s];
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid==0) && (mid < SIZE))
    d_out[blockIdx.x] = d_in[mid];
}

/* -------- KERNEL WRAPPER -------- */
void reduce(unsigned int * d_out, unsigned int * d_in, unsigned int SIZE, unsigned int NUM_THREADS)
{
  // Setting up blocks and intermediate result holder
  unsigned int NUM_BLOCKS = SIZE/NUM_THREADS + ((SIZE % NUM_THREADS)?1:0);
  unsigned int * d_intermediate_in;
  unsigned int * d_intermediate_out;
  cudaMalloc(&d_intermediate_in, sizeof(unsigned int)*SIZE);
  cudaMalloc(&d_intermediate_out, sizeof(unsigned int)*NUM_BLOCKS);
  cudaMemcpy(d_intermediate_in, d_in, sizeof(unsigned int)*SIZE, cudaMemcpyDeviceToDevice);

  // calculate shared memory
  const unsigned int SMEM = NUM_THREADS * sizeof(unsigned int);
  // Recursively solving, will run approximately log base NUM_THREADS times.
  do
  {
    shared_reduce_kernel<<<NUM_BLOCKS, NUM_THREADS, (NUM_THREADS * sizeof(unsigned int))>>>(d_intermediate_out, d_intermediate_in, SIZE);

    // Updating SIZE
    SIZE = NUM_BLOCKS;//SIZE / NUM_THREADS + SIZE_REST;

    // Updating input to intermediate
    cudaMemcpy(d_intermediate_in, d_intermediate_out, sizeof(unsigned int)*NUM_BLOCKS, cudaMemcpyDeviceToDevice);

    // Updating NUM_BLOCKS to reflect how many blocks we now want to compute on
    NUM_BLOCKS = SIZE/NUM_THREADS + ((SIZE % NUM_THREADS)?1:0);

  }
  while(SIZE > NUM_THREADS); // if it is too small, compute rest.

  // Computing rest
  shared_reduce_kernel<<<1, SIZE, SMEM>>>(d_out, d_intermediate_out, SIZE);
  cudaFree(d_intermediate_in);
  cudaFree(d_intermediate_out);
}

/* -------- MAIN -------- */
int main(int argc, char **argv)
{
  std::ofstream myfile;
  myfile.open ("par_reduce.csv");
  // Setting NUM_THREADS
  const unsigned int times = 10;
  for (unsigned int rounds = 0; rounds<30; rounds++)
  {
//    printf("Round: %d\n", rounds);
    unsigned int NUM_THREADS = 1<<10;
    // Making non-bogus data and setting it on the GPU
    unsigned int SIZE = 1<<rounds;
    unsigned int * d_in;
    unsigned int * d_out;
    cudaMalloc(&d_in, sizeof(unsigned int)*SIZE);
    cudaMalloc(&d_out, sizeof(unsigned int)*SIZE);
    unsigned int * h_in = (unsigned int *)malloc(SIZE*sizeof(int));
    for (unsigned int i = 0; i <  SIZE; i++) h_in[i] = 1;
    cudaMemcpy(d_in, h_in, sizeof(unsigned int)*SIZE, cudaMemcpyHostToDevice);

    // Running kernel wrapper
    // setting up time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel time!!!
    cudaEventRecord(start, 0);

    for (unsigned int i = 0; i < times; i++)
    {
      reduce(d_out, d_in, SIZE, NUM_THREADS);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calculating time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime = elapsedTime / ((float) times);
//    printf("time!: %.5f\n", elapsedTime);
    unsigned int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
//    printf("%d \n", h_out);
    myfile << elapsedTime << "," << std::endl;
  }
  myfile.close();
  return 0;
}
