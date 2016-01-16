// Create predicate array for HW4
#include "radix_sort.h"

/*
 * Calculate if LSB is 0.
 * 1 if true, 0 otherwise.
 */
__global__
void predicate_kernel(unsigned int* const d_predicate,
                      const unsigned int* const d_val_src,
                      const size_t numElems,
                      const unsigned int i) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_predicate[mid] = (int)(((d_val_src[mid] & (1 << i)) >> i) == 0);
}

__global__
void inclusive_sum_scan_kernel(unsigned int* const d_out,
                               const unsigned int* const d_in,
                               const int step,
                               const size_t numElems) {
  const int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

	int toAdd = (((mid - step) < 0) ? 0 : d_in[mid - step]);
  d_out[mid] = d_in[mid] + toAdd;
}

__global__
void right_shift_array_kernel(unsigned int* const d_out,
                       const unsigned int* const d_in,
                       const size_t numElems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] = (mid == 0) ? 0 : d_in[mid - 1];
}

__global__
void toggle_predicate_kernel(unsigned int* const d_out, 
                             const unsigned int* const d_predicate,
                             const size_t numElems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] = ((d_predicate[mid]) ? 0 : 1);
}

__global__
void add_splitter_map_kernel(unsigned int* const d_out,
                             const unsigned int* const shift, 
                             const size_t numElems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  d_out[mid] += shift[0];
}

__global__ 
void reduce_kernel(unsigned int* const d_out,
                   unsigned int* const d_in,
                   const size_t numElems) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
    if ((tid < s) && ((pos + s) < numElems))
      d_in[pos] = d_in[pos] + d_in[pos + s];
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid == 0) && (pos < numElems))
    d_out[blockIdx.x] = d_in[pos];
}

__global__
void map_kernel(unsigned int* const d_out,
                const unsigned int* const d_in,
                const unsigned int* const d_predicate,
                const unsigned int* const d_sum_scan_0,
                const unsigned int* const d_sum_scan_1,
                const size_t numElems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= numElems) return;

  const unsigned int pos = ((d_predicate[mid]) ? d_sum_scan_0[mid] : d_sum_scan_1[mid]);
  d_out[pos] = d_in[mid];
}

void reduce_wrapper(unsigned int* const d_out,
                    unsigned int* const d_in,
                    size_t numElems,
                    int block_size) {
  unsigned int grid_size = numElems / block_size + 1;

  unsigned int* d_tmp;
  checkCudaErrors(cudaMalloc(&d_tmp, sizeof(unsigned int) * grid_size));
  checkCudaErrors(cudaMemset(d_tmp, 0, sizeof(unsigned int) * grid_size));

  unsigned int prev_grid_size;
  unsigned int remainder = 0;
  // recursively solving, will run approximately log base block_size times.
  do {
    reduce_kernel<<<grid_size, block_size>>>(d_tmp, d_in, numElems);

    remainder = numElems % block_size;
    numElems  = numElems / block_size + remainder;

    // updating input to intermediate
    checkCudaErrors(cudaMemcpy(d_in, d_tmp, sizeof(int) * grid_size, cudaMemcpyDeviceToDevice));

    // Updating grid_size to reflect how many blocks we now want to compute on
    prev_grid_size = grid_size;
    grid_size = numElems / block_size + 1;      

    // updating intermediate
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaMalloc(&d_tmp, sizeof(int) * grid_size));
  } while(numElems > block_size);

  // computing rest
  reduce_kernel<<<1, numElems>>>(d_out, d_in, prev_grid_size);
}

void exclusive_sum_scan(unsigned int* const d_out,
                        const unsigned int* const d_predicate,
                        unsigned int* const d_predicate_tmp,
                        unsigned int* const d_sum_scan,
                        const unsigned int ARRAY_BYTES,
                        const size_t numElems,
                        const int GRID_SIZE,
                        const int BLOCK_SIZE) {
  // copy predicate values to new array
  checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  // set all elements to zero 
  checkCudaErrors(cudaMemset(d_sum_scan, 0, ARRAY_BYTES));

  // sum scan call
  for (unsigned int step = 1; step < numElems; step *= 2) {
    inclusive_sum_scan_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_sum_scan, d_predicate_tmp, step, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  }

  // shift to get exclusive scan
  checkCudaErrors(cudaMemcpy(d_out, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  right_shift_array_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_out, d_sum_scan, numElems);
}

unsigned int* radix_sort(unsigned int* h_input,
                         const size_t numElems) {
  const int BLOCK_SIZE  = 512;
  const int GRID_SIZE   = numElems / BLOCK_SIZE + 1;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * numElems;
  const unsigned int BITS_PER_BYTE = 8;

  // host memory
  unsigned int* const h_output = new unsigned int[numElems];

  // device memory
  unsigned int *d_val_src, *d_predicate, *d_sum_scan, *d_predicate_tmp, *d_sum_scan_0, *d_sum_scan_1, *d_predicate_toggle, *d_reduce, *d_map;
  checkCudaErrors(cudaMalloc((void **) &d_val_src,          ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_map,              ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate,        ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate_tmp,    ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_predicate_toggle, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan,         ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan_0,       ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_sum_scan_1,       ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_reduce, sizeof(unsigned int)));

  // copy host array to device
  checkCudaErrors(cudaMemcpy(d_val_src, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice));

  for (unsigned int i = 0; i < (BITS_PER_BYTE * sizeof(unsigned int)); i++) {
    // predicate is that LSB is 0
    predicate_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_predicate, d_val_src, numElems, i);

    // calculate scatter addresses from predicates
    exclusive_sum_scan(d_sum_scan_0, d_predicate, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, numElems, GRID_SIZE, BLOCK_SIZE);

    // copy contents of predicate, so we do not change its content
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

    // calculate how many elements had predicate equal to 1
    reduce_wrapper(d_reduce, d_predicate_tmp, numElems, BLOCK_SIZE);

    // toggle predicate values, so we can compute scatter addresses for toggled predicates
    toggle_predicate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_predicate_toggle, d_predicate, numElems);
    // so we now have addresses for elements where LSB is equal to 1
    exclusive_sum_scan(d_sum_scan_1, d_predicate_toggle, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, numElems, GRID_SIZE, BLOCK_SIZE);
    // shift scatter addresses according to amount of elements that had LSB equal to 0
    add_splitter_map_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_sum_scan_1, d_reduce, numElems);

    // move elements accordingly
    map_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_map, d_val_src, d_predicate, d_sum_scan_0, d_sum_scan_1, numElems);

    // swap pointers, instead of moving elements
    std::swap(d_val_src, d_map);
  }

  // debugging
  checkCudaErrors(cudaMemcpy(h_output, d_val_src, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  for (int i = 0; i < numElems; i++)
    printf("%u%s", h_output[i], ((i % 8 == 7) ? "\n" : "\t"));

  return h_output;
}
