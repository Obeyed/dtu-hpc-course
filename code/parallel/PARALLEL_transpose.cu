#include <stdio.h>
#include <cuda_runtime.h>
//#include "gputimer.h"

const unsigned int DIM = 12;
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

  __shared__ unsigned int tile[DIM][DIM+1]; //fucking cuda, should be blockDim.y

  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x,
               y = threadIdx.y + blockIdx.y * blockDim.y;
  if((x >= COLUMNS) || (y >= ROWS))
  {
//  	printf("RETURNED: %d, %d \n", x, y);
    return;
  }
  tile[threadIdx.y][threadIdx.x] = d_in[x + y*COLUMNS];
//  printf("FIRST: id: threadIdx[y=%d][x=%d]=%d\n",threadIdx.y, threadIdx.x, tile[threadIdx.y][threadIdx.x]);
//  printf("from in_position: %d\n", x + y*COLUMNS);
  __syncthreads();

  x = threadIdx.x + blockIdx.y * blockDim.y;
  y = threadIdx.y + blockIdx.x * blockDim.x;

  d_out[x + y*ROWS] = tile[threadIdx.x][threadIdx.y];
//  printf("I am tile[%d][%d] = %d \n", x, y, tile[threadIdx.x][threadIdx.y]);
//  printf("to out_position %d\n", x + y*ROWS);
}




int main(int argc, char **argv){
  const unsigned int ROWS = 1<<10,
                     COLUMNS = 1<<10,
                     BYTES_ARRAY = ROWS*COLUMNS*sizeof(unsigned int);

  unsigned int * h_in = (unsigned int *) malloc(BYTES_ARRAY),
               * h_out = (unsigned int *) malloc(BYTES_ARRAY),
               * gold = (unsigned int *) malloc(BYTES_ARRAY);

  fill_matrix(h_in, ROWS, COLUMNS);
  transpose_CPU(h_in, gold, ROWS, COLUMNS);

  unsigned int * d_in, * d_out;

  cudaMalloc(&d_in, BYTES_ARRAY);
  cudaMalloc(&d_out, BYTES_ARRAY);
  cudaMemcpy(d_in, h_in, BYTES_ARRAY, cudaMemcpyHostToDevice);

//  GpuTimer timer;

  /* STARTING KERNEL */

  const dim3 GRID_SIZE(ROWS/BLOCK_SIZE.x + 1, COLUMNS/BLOCK_SIZE.y + 1);
//  timer.start();
//  transpose_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, ROWS, COLUMNS);
  transpose_kernel_tiled<<<GRID_SIZE, BLOCK_SIZE, DIM*DIM*sizeof(unsigned int)+1>>>(d_out, d_in, ROWS, COLUMNS);
//  timer.stop();

  cudaMemcpy(h_out, d_out, BYTES_ARRAY, cudaMemcpyDeviceToHost);
  printf("transpose_serial\nVerifying transpose...%s\n", 
           compare_matrices(h_out, gold, ROWS, COLUMNS) ? "Failed" : "Success");

  cudaFree(d_in);
  cudaFree(d_out);
}