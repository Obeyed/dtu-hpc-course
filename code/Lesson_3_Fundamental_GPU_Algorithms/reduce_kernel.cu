#include <stdio.h>

/* -------- KERNEL -------- */
__global__ void reduce_kernel(float * d_out, float * d_in, const int size)
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
void reduce(float * d_out, float * d_in, int size, int num_threads)
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
  float * d_intermediate;
  cudaMalloc(&d_intermediate, sizeof(float)*num_blocks);
  cudaMemset(d_intermediate, 0, sizeof(float)*num_blocks);
  int prev_num_blocks;
  int i = 1;
  // recursively solving, will run approximately log base num_threads times.
  do
  {
    printf("Round:%.d\n", i);
    printf("NumBlocks:%.d\n", num_blocks);
    printf("NumThreads:%.d\n", num_threads);
    i++;
    reduce_kernel<<<num_blocks, num_threads>>>(d_intermediate, d_in, size);

    // updating input to intermediate
    cudaMemcpy(d_in, d_intermediate, sizeof(float)*num_blocks, cudaMemcpyDeviceToDevice);

    // Updating num_blocks to reflect how many blocks we now want to compute on
    prev_num_blocks = num_blocks;
    if(num_blocks % num_threads)
    {
      num_blocks = num_blocks / num_threads + 1;      
    }
    else
    {
      num_blocks = num_blocks / num_threads;
    }


    // updating intermediate
    cudaFree(d_intermediate);
    cudaMalloc(&d_intermediate, sizeof(float)*num_blocks);
    size = num_blocks*num_threads;
  }
  while(prev_num_blocks > num_threads); // if it is too small, compute rest.

  // computing rest
  reduce_kernel<<<1, prev_num_blocks>>>(d_out, d_in, prev_num_blocks);

}

/* -------- MAIN -------- */
int main(int argc, char **argv)
{
  // Setting num_threads
  int num_threads = 16;
  // Making non-bogus data and setting it on the GPU
  const int size = 16384;
  const int size_out = 1;
  float * d_in;
  float * d_out;
  cudaMalloc(&d_in, sizeof(float)*size);
  cudaMalloc(&d_out, sizeof(float)*size_out);
  //const int value = 5;
  //cudaMemset(d_in, value, sizeof(float)*size);
  float * h_in = (float *)malloc(size*sizeof(float));
  for (int i = 0; i <  size; i++) h_in[i] = 1.0f;
  cudaMemcpy(d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);

  // Running kernel wrapper
  reduce(d_out, d_in, size, num_threads);
  float result;
  cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  printf("sum is element is: %.f\n", result);
}