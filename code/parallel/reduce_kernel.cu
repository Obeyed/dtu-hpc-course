#include <stdio.h>

/* -------- KERNEL -------- */
__global__ void reduce_kernel(int * d_out, int * d_in, int size)
{
  // position and threadId
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // do reduction in global memory
  for (unsigned int s = blockDim.x / 2; s>0; s>>=1)
  {
    if (tid < s)
    {
      if (pos+s < size) // Handling out of bounds
      {
        d_in[pos] = d_in[pos] + d_in[pos+s];
      }
    }
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid==0) && (pos < size))
  {
    d_out[blockIdx.x] = d_in[pos];
  }
}

/* -------- KERNEL WRAPPER -------- */
void reduce(int * d_out, int * d_in, int size, int num_threads)
{
  // setting up blocks and intermediate result holder
  
  int num_blocks;
  if(((size) % num_threads))
    {
      num_blocks = ((size) / num_threads) + 1;    
    }
    else
    {
      num_blocks = (size) / num_threads;
    }
  int * d_intermediate;
  cudaMalloc(&d_intermediate, sizeof(int)*num_blocks);
  cudaMemset(d_intermediate, 0, sizeof(int)*num_blocks);
  int prev_num_blocks;
  int i = 1;
  int size_rest = 0;
  // recursively solving, will run approximately log base num_threads times.
  do
  {
    printf("Round:%.d\n", i);
    printf("NumBlocks:%.d\n", num_blocks);
    printf("NumThreads:%.d\n", num_threads);
    printf("size of array:%.d\n", size);
    i++;
    reduce_kernel<<<num_blocks, num_threads>>>(d_intermediate, d_in, size);
    size_rest = size % num_threads;
    size = size / num_threads + size_rest;

    // updating input to intermediate
    cudaMemcpy(d_in, d_intermediate, sizeof(int)*num_blocks, cudaMemcpyDeviceToDevice);

    // Updating num_blocks to reflect how many blocks we now want to compute on
    prev_num_blocks = num_blocks;
    if(size % num_threads)
    {
      num_blocks = size / num_threads + 1;      
    }
    else
    {
      num_blocks = size / num_threads;
    }
    // updating intermediate
    cudaFree(d_intermediate);
    cudaMalloc(&d_intermediate, sizeof(int)*num_blocks);
  }
  while(size > num_threads); // if it is too small, compute rest.

  // computing rest
  reduce_kernel<<<1, size>>>(d_out, d_in, prev_num_blocks);

}

/* -------- MAIN -------- */
int main(int argc, char **argv)
{
  printf("@@STARTING@@ \n");
  // Setting num_threads
  int num_threads = 512;
  // Making non-bogus data and setting it on the GPU
  const int size = 1<<19;
  const int size_out = 1;
  int * d_in;
  int * d_out;
  cudaMalloc(&d_in, sizeof(int)*size);
  cudaMalloc(&d_out, sizeof(int)*size_out);

  int * h_in = (int *)malloc(size*sizeof(int));
  for (int i = 0; i <  size; i++) h_in[i] = 1;
  cudaMemcpy(d_in, h_in, sizeof(int)*size, cudaMemcpyHostToDevice);

  // Running kernel wrapper
  reduce(d_out, d_in, size, num_threads);
  int result;
  cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("\nFINAL SUM IS: %d\n", result);
}