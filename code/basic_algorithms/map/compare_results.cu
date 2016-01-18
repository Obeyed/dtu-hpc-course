#include "par_mapping.h"
#include "utils.h"
#include "ser_mapping.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  // change
  const size_t num_elems = 1 << 5;
  // define following as needed
  unsigned int* input_vals    = new unsigned int[num_elems];
  unsigned int* serial_output = new unsigned int[num_elems];

  // define initial values
  printf("INITIALISING RANDOM VALUES..\n");
  srand(time(NULL));
  for (unsigned int i = 0; i < num_elems; i++) {
    input_vals[i] = rand(); 
  }

  // call parallel code
  printf("CALLING GPU CODE..\n");
  unsigned int* parallel_output = par_map(input_vals, num_elems);

  // call serial code
  printf("CALLING REFERENCE..\n");
  ser_map(&input_vals[0],  &input_pos[0],
          &serial_output[0], &output_pos[0],
          num_elems);

  // compare results
  printf("COMPARING RESULTS..\n");
  checkResultsExact(&serial_output[0], &parallel_output[0], num_elems);

  // if this is reached, then success
  printf("SUCCESS!\n");

  return 0;
}
