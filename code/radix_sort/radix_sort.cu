/**
    High Performance Computing (special course)
    radix_sort.cu
    Location: Technical University of Denmark
    Purpose: Uses GPU to sort series of unsigned integers using Radix Sort

    @author Elias Obeid
    @author Alexander Johansen
    @version 1.0 16/01/2016
*/

#include "radix_sort.h"

/**
    Populates array with 1/0 depending on Least Significant Bit is set.
    If LSB is 0 then index is set to 1, otherwise 0.

    @param d_predicate  Output array to be filled with values (predicates)
    @param d_val_src    Values to run through
    @param num_elems     Number of elements in arrays
    @param i            Used to calculate how much to shift to find the correct LSB
*/
__global__ 
void predicate_kernel(unsigned int* const, 
                      const unsigned int* const,
                      const size_t,
                      const unsigned int);
/**
    Performs one iteration of Hillis and Steele scan.
    Inclusive sum scan.

    @param d_out    Output array with summed values
    @param d_in     Values to sum
    @param step     Amount to look back in d_in
    @param num_elems Number of elements in arrays
*/
__global__
void inclusive_sum_scan_kernel(unsigned int* const,
                               const unsigned int* const,
                               const int,
                               const size_t);
/**
    Shifts all elements to the right.
    Sets first index to 0.

    @param d_out    Output array
    @param d_in     Array to be shifted
    @param num_elems Number of elements in arrays
*/
__global__
void right_shift_array_kernel(unsigned int* const,
                              const unsigned int* const,
                              const size_t);
/**
    Toggle array with values 1 and 0.

    @param d_out        Array with toggled values
    @param d_predicate  Array with initial values
    @param num_elems     Number of elements in arrays
*/
__global__
void toggle_predicate_kernel(unsigned int* const, 
                             const unsigned int* const,
                             const size_t);
/**
    Adds an offset to the given array's values.

    @param d_out      Input/Output array -- values will be added to offset
    @param shift      Array with one element -- the offset to add
    @param num_elems   Number of elements in arrays
*/
__global__
void add_splitter_map_kernel(unsigned int* const,
                             const unsigned int* const, 
                             const size_t);
/**
    Runs log_2(BLOCK_SIZE) iterations of the reduce.
    Computes the sum of elements in d_in

    @param d_out     Output array
    @param d_in      Input array with values
    @param num_elems  Number of elements in arrays
*/
__global__ 
void reduce_kernel(unsigned int* const,
                   unsigned int* const,
                   const size_t);
/**
    Maps values from d_in to d_out according to scatter addresses in d_sum_scan_0 or d_sum_scan_1.

    @param d_out        Output array
    @param d_in         Input array with values
    @param d_predicate  Contains whether or not given value's LSB is 0
    @param d_sum_scan_0 Scatter address for values with LSB 0
    @param d_sum_scan_1 Scatter address for values with LSB 1
    @param num_elems     Number of elements in arrays
*/
__global__
void map_kernel(unsigned int* const,
                const unsigned int* const,
                const unsigned int* const,
                const unsigned int* const,
                const unsigned int* const,
                const size_t);
/**
    Calls reduce kernel to compute reduction.
    Runs log_(BLOCK_SIZE)(num_elems) times.

    @param d_out      Output array
    @param d_in       Input array with values
    @param num_elems   Number of elements in arrays
    @param block_size Number of threads per block
*/
void reduce_wrapper(unsigned int* const,
                    unsigned int* const,
                    size_t,
                    int);
/**
    Computes an exclusive sum scan of scatter addresses for the given predicate array.

    @param d_out            Output array with scatter addresses
    @param d_predicate      Input array with predicates to be summed
    @param d_predicate_tmp  Temporary array so we do not change d_predicate
    @param d_sum_scan       Inclusive sum scan
    @param ARRAY_BYTES      Number of bytes for arrays
    @param num_elems         Number of elements in arrays
    @param GRID_SIZE        Number of blocks in one grid
    @param BLOCK_SIZE       Number of threads in one block
*/
void exclusive_sum_scan(unsigned int* const,
                        const unsigned int* const,
                        unsigned int* const,
                        unsigned int* const,
                        const unsigned int,
                        const size_t,
                        const int,
                        const int);
/**
    Sort values using radix sort.

    @param h_input  Input values to be sorted (unsigned int)
    @param num_elems Number of elements in array
    @return Pointer to sorted array
*/
unsigned int* radix_sort(unsigned int*,
                         const size_t);
// Populates array with 1/0 depending on Least Significant Bit is set.
__global__
void predicate_kernel(unsigned int* const d_predicate,
                      const unsigned int* const d_val_src,
                      const size_t num_elems,
                      const unsigned int i) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= num_elems) return;

  d_predicate[mid] = (int)(((d_val_src[mid] & (1 << i)) >> i) == 0);
}

// Performs one iteration of Hillis and Steele scan.
__global__
void inclusive_sum_scan_kernel(unsigned int* const d_out,
                               const unsigned int* const d_in,
                               const int step,
                               const size_t num_elems) {
  const int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= num_elems) return;

	int toAdd = (((mid - step) < 0) ? 0 : d_in[mid - step]);
  d_out[mid] = d_in[mid] + toAdd;
}

// Shifts all elements to the right. Sets first index to 0.
__global__
void right_shift_array_kernel(unsigned int* const d_out,
                       const unsigned int* const d_in,
                       const size_t num_elems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= num_elems) return;

  d_out[mid] = (mid == 0) ? 0 : d_in[mid - 1];
}

// Toggle array with values 1 and 0.
__global__
void toggle_predicate_kernel(unsigned int* const d_out, 
                             const unsigned int* const d_predicate,
                             const size_t num_elems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= num_elems) return;

  d_out[mid] = ((d_predicate[mid]) ? 0 : 1);
}

// Adds an offset to the given array's values.
__global__
void add_splitter_map_kernel(unsigned int* const d_out,
                             const unsigned int* const shift, 
                             const size_t num_elems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= num_elems) return;

  d_out[mid] += shift[0];
}

// Computes the sum of elements in d_in
__global__ 
void reduce_kernel(unsigned int* const d_out,
                   unsigned int* const d_in,
                   const size_t num_elems) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
    if ((tid < s) && ((pos + s) < num_elems))
      d_in[pos] = d_in[pos] + d_in[pos + s];
    __syncthreads();
  }

  // only thread 0 writes result, as thread
  if ((tid == 0) && (pos < num_elems))
    d_out[blockIdx.x] = d_in[pos];
}

// Maps values from d_in to d_out according to scatter addresses in d_sum_scan_0 or d_sum_scan_1.
__global__
void map_kernel(unsigned int* const d_out,
                const unsigned int* const d_in,
                const unsigned int* const d_predicate,
                const unsigned int* const d_sum_scan_0,
                const unsigned int* const d_sum_scan_1,
                const size_t num_elems) {
  const unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid >= num_elems) return;

  const unsigned int pos = ((d_predicate[mid]) ? d_sum_scan_0[mid] : d_sum_scan_1[mid]);
  d_out[pos] = d_in[mid];
}

// Calls reduce kernel to compute reduction.
void reduce_wrapper(unsigned int* const d_out,
                    unsigned int* const d_in,
                    size_t num_elems,
                    int block_size) {
  unsigned int grid_size = num_elems / block_size + 1;

  unsigned int* d_tmp;
  checkCudaErrors(cudaMalloc(&d_tmp, sizeof(unsigned int) * grid_size));
  checkCudaErrors(cudaMemset(d_tmp, 0, sizeof(unsigned int) * grid_size));

  unsigned int prev_grid_size;
  unsigned int remainder = 0;
  // recursively solving, will run approximately log base block_size times.
  do {
    reduce_kernel<<<grid_size, block_size>>>(d_tmp, d_in, num_elems);

    remainder = num_elems % block_size;
    num_elems  = num_elems / block_size + remainder;

    // updating input to intermediate
    checkCudaErrors(cudaMemcpy(d_in, d_tmp, sizeof(int) * grid_size, cudaMemcpyDeviceToDevice));

    // Updating grid_size to reflect how many blocks we now want to compute on
    prev_grid_size = grid_size;
    grid_size = num_elems / block_size + 1;      

    // updating intermediate
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaMalloc(&d_tmp, sizeof(int) * grid_size));
  } while(num_elems > block_size);

  // computing rest
  reduce_kernel<<<1, num_elems>>>(d_out, d_in, prev_grid_size);
}

// Computes an exclusive sum scan of scatter addresses for the given predicate array.
void exclusive_sum_scan(unsigned int* const d_out,
                        const unsigned int* const d_predicate,
                        unsigned int* const d_predicate_tmp,
                        unsigned int* const d_sum_scan,
                        const unsigned int ARRAY_BYTES,
                        const size_t num_elems,
                        const int GRID_SIZE,
                        const int BLOCK_SIZE) {
  // copy predicate values to new array
  checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  // set all elements to zero 
  checkCudaErrors(cudaMemset(d_sum_scan, 0, ARRAY_BYTES));

  // sum scan call
  for (unsigned int step = 1; step < num_elems; step *= 2) {
    inclusive_sum_scan_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_sum_scan, d_predicate_tmp, step, num_elems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  }

  // shift to get exclusive scan
  checkCudaErrors(cudaMemcpy(d_out, d_sum_scan, ARRAY_BYTES, cudaMemcpyDeviceToDevice));
  right_shift_array_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_out, d_sum_scan, num_elems);
}

// Sort values using radix sort
unsigned int* radix_sort(unsigned int* h_input,
                         const size_t num_elems) {
  const int BLOCK_SIZE  = 512;
  const int GRID_SIZE   = num_elems / BLOCK_SIZE + 1;
  const unsigned int ARRAY_BYTES = sizeof(unsigned int) * num_elems;
  const unsigned int BITS_PER_BYTE = 8;

  // host memory
  unsigned int* const h_output = new unsigned int[num_elems];

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
    predicate_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_predicate, d_val_src, num_elems, i);

    // calculate scatter addresses from predicates
    exclusive_sum_scan(d_sum_scan_0, d_predicate, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, num_elems, GRID_SIZE, BLOCK_SIZE);

    // copy contents of predicate, so we do not change its content
    checkCudaErrors(cudaMemcpy(d_predicate_tmp, d_predicate, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

    // calculate how many elements had predicate equal to 1
    reduce_wrapper(d_reduce, d_predicate_tmp, num_elems, BLOCK_SIZE);

    // toggle predicate values, so we can compute scatter addresses for toggled predicates
    toggle_predicate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_predicate_toggle, d_predicate, num_elems);
    // so we now have addresses for elements where LSB is equal to 1
    exclusive_sum_scan(d_sum_scan_1, d_predicate_toggle, d_predicate_tmp, d_sum_scan, ARRAY_BYTES, num_elems, GRID_SIZE, BLOCK_SIZE);
    // shift scatter addresses according to amount of elements that had LSB equal to 0
    add_splitter_map_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_sum_scan_1, d_reduce, num_elems);

    // move elements accordingly
    map_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(d_map, d_val_src, d_predicate, d_sum_scan_0, d_sum_scan_1, num_elems);

    // swap pointers, instead of moving elements
    std::swap(d_val_src, d_map);
  }

  // copy contents back
  checkCudaErrors(cudaMemcpy(h_output, d_val_src, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  return h_output;
}
