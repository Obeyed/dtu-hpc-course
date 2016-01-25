#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

const unsigned int DIM = 32;
const dim3 BLOCK_SIZE(DIM, DIM);

// For testing
unsigned int compare_matrices(unsigned int *gpu, unsigned int *ref,
                              const unsigned int ROWS, const unsigned int COLUMNS){
  unsigned int result = 0;

  for(unsigned int i=0; i < COLUMNS; i++)
    for(unsigned int j=0; j < ROWS; j++)
      if (ref[i + j*COLUMNS] != gpu[i + j*COLUMNS]){
        //printf("reference(%d,%d) = %d but test(%d,%d) = %d\n",
        //       i,j,ref[i+j*COLUMNS],i,j,gpu[i+j*COLUMNS]);
        result = 1;
      }else{
        //printf("WORKED: reference(%d,%d) = %d test(%d,%d) = %d\n",
        //i,j,ref[i+j*COLUMNS],i,j,gpu[i+j*COLUMNS]);
      }
  return result;
}

void fill_matrix(unsigned int * mat,
	             const unsigned int ROWS, const unsigned int COLUMNS){
	for(unsigned int i=0; i < ROWS * COLUMNS; i++)
		mat[i] = (unsigned int) i;
}

/* CPU KERNEL */
void transpose_CPU(unsigned int * in, unsigned int * out,
                   const unsigned int ROWS, const unsigned int COLUMNS){
	for(unsigned int row=0; row < ROWS; row++)
    	for(unsigned int column=0; column < COLUMNS; column++)
      		out[column + row*COLUMNS] = in[row + column*ROWS]; // out(j,i) = in(i,j)
}

/* KERNEL */
__global__
void transpose_kernel(unsigned int * d_out, unsigned int * d_in,
                      const unsigned int ROWS, const unsigned int COLUMNS){
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int column = threadIdx.y + blockIdx.y * blockDim.y;
  if((row >= ROWS) || (column >= COLUMNS)) return;
  d_out[column + row*COLUMNS] = d_in[row + column*ROWS];
}

__global__
void transpose_kernel_tiled(unsigned int * d_out, unsigned int * d_in,
                            const unsigned int ROWS, const unsigned int COLUMNS){
  __shared__ unsigned int tile[DIM][DIM]; 
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x,
               y = threadIdx.y + blockIdx.y * blockDim.y;
  if((x >= COLUMNS) || (y >= ROWS)) return;
  tile[threadIdx.y][threadIdx.x] = d_in[x + y*COLUMNS];
  __syncthreads();
  x = threadIdx.x + blockIdx.y * blockDim.y;
  y = threadIdx.y + blockIdx.x * blockDim.x;
  d_out[x + y*ROWS] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char **argv){
  unsigned int times = 1;
  printf("Starting!\n");
  const unsigned int ROWS = 1<<14,
                     COLUMNS = 1<<14,
                     BYTES_ARRAY = ROWS*COLUMNS*sizeof(unsigned int);
  printf("Bytes sat\n");
  unsigned int * h_in = (unsigned int *) malloc(BYTES_ARRAY),
               * h_out = (unsigned int *) malloc(BYTES_ARRAY),
               * gold = (unsigned int *) malloc(BYTES_ARRAY);
  printf("pointers sat\n");
  printf("Filling matrix\n");
  fill_matrix(h_in, ROWS, COLUMNS);
  printf("Transposing!\n");


  unsigned int * d_in, * d_out;

  cudaMalloc(&d_in, BYTES_ARRAY);
  cudaMalloc(&d_out, BYTES_ARRAY);
  cudaMemcpy(d_in, h_in, BYTES_ARRAY, cudaMemcpyHostToDevice);

//  GpuTimer timer;

  /* STARTING KERNEL */

  const dim3 GRID_SIZE(ROWS/BLOCK_SIZE.x + 1, COLUMNS/BLOCK_SIZE.y + 1);
transpose_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, ROWS, COLUMNS);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

//  for (unsigned int k = 0; k<times; k++){
  //  transpose_kernel_tiled<<<GRID_SIZE, BLOCK_SIZE, DIM*(DIM)*sizeof(unsigned int)>>>(d_out, d_in, ROWS, COLUMNS);
  transpose_CPU(h_in, gold, ROWS, COLUMNS);
//  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  // calculating time
  float elapsedTime = .0f;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  elapsedTime = elapsedTime / ((float) times);
  printf(" time: %.5f\n", elapsedTime);

  cudaMemcpy(h_out, d_out, BYTES_ARRAY, cudaMemcpyDeviceToHost);
  printf("transpose_serial\nVerifying transpose...%s\n", 
           compare_matrices(h_out, gold, ROWS, COLUMNS) ? "Failed" : "Success");

  cudaFree(d_in);
  cudaFree(d_out);
}