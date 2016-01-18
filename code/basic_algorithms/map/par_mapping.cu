#include <par_map.h>
#include <cuda_runtime.h>

__global__ 
void mapping_kernel(unsigned int* const d_out, 
                    unsigned int* const d_in, 
                    const unsigned int IN_SIZE) {
	const unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId >= IN_SIZE) return;
	d_out[myId] = d_in[myId];
}

int main(int argc, char **argv)
{
		const unsigned int IN_SIZE = 1<<rounds;
		const unsigned int IN_BYTES = sizeof(unsigned int) * IN_SIZE;
		const unsigned int OUT_SIZE = IN_SIZE;
		const unsigned int OUT_BYTES = IN_BYTES;
		const dim3 NUM_THREADS(1<<10);
		const dim3 NUM_BLOCKS(IN_SIZE/NUM_THREADS.x + ((IN_SIZE % NUM_THREADS.x)?1:0));

		// Generate the input array on host
		unsigned int * h_in = new unsigned int[IN_SIZE];
		unsigned int * h_out = new unsigned int[OUT_SIZE];

		// Declare GPU memory pointers
		unsigned int * d_in;
		unsigned int * d_out;

		// Allocate GPU memory
		cudaMalloc((void **) &d_in, IN_BYTES);
		cudaMalloc((void **) &d_out, OUT_BYTES);

		// Transfer the arrays to the GPU
		cudaMemcpy(d_in, h_in, IN_BYTES, cudaMemcpyHostToDevice);

            mapping_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in, IN_SIZE);
		cudaMemcpy(h_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost);
		cudaFree(d_in);
		cudaFree(d_out);
	}
}
