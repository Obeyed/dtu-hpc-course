__global__ void reduce_kernel(float * d_out, float * d_in)
{
  // position and threadId
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // do reduction in global memory
  for (unsigned int s = blockDim.x / 2; s>0; s>>=1)
  {
    if (tid < s)
    {
      d_in[pos] = op(d_in[pos], d_in[pos+s]);
    }
  }

  // only thread 0 writes result, as thread
  if (tid==0)
  {
    d_out[blockIdx.x] = d_in[pos];
  }
}

void reduce(float * d_out, float * d_in, const int size, int num_threads)
{
  // need to have at least two threads for the kernel to function
  assert(num_threads >= 2);

  // setting up blocks and intermediate result holder
  int num_blocks = ((size) / num_threads) + 1;
  float * d_intermediate;
  checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float)*num_blocks));

  // recursively solving, will run approximately log base num_threads times.
  do
  {
    reduce_kernel<<<num_blocks, num_threads>>>(d_intermediate, d_in);
    // updating input to intermediate
    checkCudaErrors(cudaMemcpy(d_in, d_intermediate, sizeof(float)*num_blocks));
    // updating num_blocks to reflect how many blocks we now want to compute on
    num_blocks = num_blocks / num_threads + 1;
    // updating intermediate
    checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float)*num_blocks));
  }
  while(num_blocks > num_threads); // if it is too small, we just compute it on as the rest.

  // computing rest
  reduce_kernel<<<1, num_blocks>>>(d_out, d_in);

}

int main(int argc, char **argv)
{
  // Setting num_threads
  int num_threads = 512;
  // Setting operator
  float op = min();

  // Setting fake data on GPU
  const int size = 100000;
  float* d_in, d_out;
  checkCudaErrors(cudaMalloc(d_in, sizeof(float)*size));
  checkCudaErrors(cudaMalloc(d_out, sizeof(float)));
  const int value = 5;
  cudaMemset(d_in, value, sizeof(float)*size);

  // Running kernel wrapper
  reduce(d_out, d_in, size, num_threads);

  printf("min element is: %.f", d_out[0]);

}
