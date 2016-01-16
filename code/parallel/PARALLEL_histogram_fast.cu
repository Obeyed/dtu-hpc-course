#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

const unsigned int BLOCK_SIZE = 1024;

__global__
void fastHisto_kernel(unsigned int ** d_out,
               unsigned int * d_in,
               unsigned int SIZE){
  unsigned int mid = threadIdx.x + blockIdx.x*blockDim.x;
  if(mid>=SIZE) return;
  unsigned int myVal = d_in[mid];
  atomicAdd(&(d_out[blockIdx.x][myVal]),1);
}

__global__
void fill_pointers_kernel(unsigned int * d_in, unsigned int SIZE, unsigned int rows){
  unsigned int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if(mid>=SIZE) return;
  d_in[mid] = d_in[0] + rows*mid;
}

__global__
void transpose_kernel(unsigned int ** d_out, unsigned int ** d_in, unsigned int GRID_SIZE, unsigned int OUT_SIZE){
  for(int j=0; j < OUT_SIZE; j++)
    for(int i=0; i < GRID_SIZE; i++)
      d_out[j][i] = d_in[i][j]; // out(j,i) = in(i,j)
}

/* -------- KERNEL -------- */
__global__
void reduce_kernel(unsigned int * d_out, unsigned int * d_in, unsigned int SIZE, unsigned int bin, bool last)
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
  	if(last==false)
      d_out[blockIdx.x] = d_in[mid];
    else
      d_out[bin] = d_in[mid];
}

/* -------- REDUCE KERNEL WRAPPER -------- */
void reduce(unsigned int * d_out, unsigned int * d_in, unsigned int SIZE, unsigned int bin)
{
  // Setting up blocks and intermediate result holder
  unsigned int SIZE_REDUCE = SIZE;
  unsigned int GRID_SIZE_REDUCE = SIZE/BLOCK_SIZE + ((SIZE % BLOCK_SIZE)?1:0);
  unsigned int * d_intermediate;
  cudaMalloc(&d_intermediate, sizeof(unsigned int)*GRID_SIZE_REDUCE);
  // Recursively solving, will run approximately log base BLOCK_SIZE times.
  do
  {
    reduce_kernel<<<GRID_SIZE_REDUCE, BLOCK_SIZE>>>(d_intermediate, d_in, SIZE_REDUCE, false, bin);

    // Updating SIZE
    SIZE_REDUCE = GRID_SIZE_REDUCE;//SIZE / NUM_THREADS + SIZE_REST;

    // Updating input to intermediate
    std::swap(d_in, d_intermediate);

    // Updating NUM_BLOCKS to reflect how many blocks we now want to compute on
    GRID_SIZE_REDUCE = SIZE_REDUCE/BLOCK_SIZE + ((SIZE_REDUCE % BLOCK_SIZE)?1:0);
  }
  while(SIZE_REDUCE > BLOCK_SIZE); // if it is too small, compute rest.

  // Computing rest
  reduce_kernel<<<1, SIZE>>>(d_out, d_in, SIZE_REDUCE, true, bin);
  cudaFree(d_intermediate);
}

void merge(unsigned int * d_out, unsigned int * d_in, unsigned int SIZE, unsigned int bin){
  reduce(d_out, d_in, SIZE, bin);
}

void fastHisto(unsigned int * d_out, unsigned int * d_in, unsigned int IN_SIZE, unsigned int GRID_SIZE, unsigned int OUT_SIZE){
  //Setting up major histo
  unsigned int ** d_out_all, ** d_out_all_trans;
  unsigned int GRID_SIZE_ALL = GRID_SIZE * OUT_SIZE;
  unsigned int GRID_BYTES_ALL = GRID_SIZE_ALL * sizeof(unsigned int);
  cudaMalloc(&d_out_all, GRID_BYTES_ALL);
  cudaMemset(d_out_all, 0, GRID_BYTES_ALL);
  fastHisto_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out_all, d_in, IN_SIZE);
  cudaMalloc(&d_out_all_trans, GRID_BYTES_ALL);
  transpose_kernel<<<1, 1>>>(d_out_all_trans, d_out_all, GRID_SIZE, OUT_SIZE);
  cudaFree(d_out_all);

  // Merging histograms reduce
  for(unsigned int bin = 0; bin<OUT_SIZE; bin++){
    merge(d_out, d_out_all_trans[bin], GRID_SIZE, bin);
  }
  cudaFree(d_out_all_trans);
}

int main(int argc, char **argv){
  printf("---STARTED---\n");
  // Vars
  unsigned int IN_SIZE;
  unsigned int IN_BYTES;
  unsigned int OUT_SIZE;
  unsigned int OUT_BYTES;
  unsigned int GRID_SIZE;
  unsigned int h_filler;
  unsigned int sum;

  for(unsigned int rounds = 2; rounds<8; rounds++){
    IN_SIZE = 1<<8;
    IN_BYTES = sizeof(unsigned int) * IN_SIZE;
    OUT_SIZE = 1<<rounds;
    OUT_BYTES = sizeof(unsigned int) * OUT_SIZE;
    GRID_SIZE = IN_SIZE/BLOCK_SIZE + ((IN_SIZE % BLOCK_SIZE)?1:0);

    // Generate the input array on host
    unsigned int * h_in = (unsigned int *)malloc(IN_BYTES);
    unsigned int * h_out = (unsigned int *)malloc(OUT_BYTES);
    for (h_filler = 0; h_filler<IN_SIZE; h_filler++) {h_in[h_filler] = h_filler;}

    // Declare GPU memory pointers
    printf("\n@@@ROUND@@@: %d\n", rounds);
    printf("---IN_SIZE---: %d\n", IN_SIZE);
    printf("---IN_BYTES---: %d\n", IN_BYTES);
    printf("---OUT_SIZE---: %d\n", OUT_SIZE);
    printf("---OUT_BYTES---: %d\n", OUT_BYTES);
    printf("---BLOCK_SIZE---: %d\n", BLOCK_SIZE);
    printf("---GRID_SIZE---: %d\n", GRID_SIZE);

    unsigned * d_in;
	unsigned * d_out;
	// Allocate GPU memory
    cudaMalloc(&d_in, IN_BYTES);
    printf("---ALLOCATED D_IN---\n");
    cudaMalloc(&d_out, OUT_BYTES);
    printf("---ALLOCATED D_OUT---\n");

    // Transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, IN_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // running the code on the GPU
    cudaMemset(d_out, 0, OUT_BYTES);
    fastHisto(d_out, d_in, IN_SIZE, GRID_SIZE, OUT_SIZE);
//    simple_histo<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, OUT_SIZE, IN_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculating time
    float elapsedTime = .0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //elapsedTime = elapsedTime / ((float) times);
    printf(" time: %.5f\n", elapsedTime);

    // Copy back to HOST
    cudaMemcpy(h_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost);
    sum = 0;
    for(unsigned int i = 0; i<OUT_SIZE; i++) sum += h_out[i];
    for(unsigned int i = 0; (i<OUT_SIZE) && (i<10); i++){
      printf("bin %d: count %d\n", i, h_out[i]);
    }
    printf("%d\n", sum);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
  }
  return 0;
}