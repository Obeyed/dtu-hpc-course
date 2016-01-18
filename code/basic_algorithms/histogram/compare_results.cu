#include "par_scan.h"
#include "reference_calc.h"
#include "utils.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  // change
  const size_t num_elems = 1 << 5;
  const size_t BIN_SIZE  = 1 << 2;
  // define following as needed
  unsigned int* input_vals      = new unsigned int[num_elems];
  unsigned int* parallel_output = new unsigned int[num_elems];
  unsigned int* serial_output   = new unsigned int[num_elems];

  // define initial values
  printf("INITIALISING RANDOM VALUES..\n");
  srand(time(NULL));
  for (unsigned int i = 0; i < num_elems; i++) {
    input_vals[i] = rand(); 
  }

  // call parallel code
  printf("CALLING GPU CODE..\n");
  par_histogram(parallel_output, input_vals, num_elems, BIN_SIZE);

  // call serial code
  printf("CALLING REFERENCE..\n");
  reference_calculation(serial_output, input_vals, num_elems, BIN_SIZE);

  // compare results
  printf("COMPARING RESULTS..\n");
  checkResultsExact(&serial_output[0], &parallel_output[0], 1);

  // if this is reached, then success
  printf("SUCCESS!\n");

  return 0;
}
