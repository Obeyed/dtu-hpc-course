#include <stdio.h>
#include "gputimer.h"

int NUM_THREADS = 1000000;
int ARRAY_SIZE = 10;

int BLOCK_WIDTH = 1000;

void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
    printf("}\n");
}

__global__ void increment_naive(int *g, int *as)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % as[0];
	g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g, int *as)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % as[0];
	atomicAdd(& g[i], 1);
}

int main(int argc,char **argv)
{   
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
 
    // declare, allocate, and zero out GPU memory
    int * d_array;
    int * d_ARRAY_SIZE;
    int * h_ARRAY_SIZE = &ARRAY_SIZE;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMalloc((void **) &d_ARRAY_SIZE, sizeof(int));
    
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 
    cudaMemcpy(d_ARRAY_SIZE, h_ARRAY_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    // launch the kernel - comment out one of these
    timer.Start();
    
    // Instructions: This program is needed for the next quiz
    // uncomment increment_naive to measure speed and accuracy 
    // of non-atomic increments or uncomment increment_atomic to
    // measure speed and accuracy of  atomic icrements
    // increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array, d_ARRAY_SIZE);
    timer.Stop();
    
    // copy back the array of sums from GPU and print
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed = %g ms\n", timer.Elapsed());
 
    // free GPU memory allocation and exit
    cudaFree(d_array);
    cudaFree(d_ARRAY_SIZE);
    return 0;
}
