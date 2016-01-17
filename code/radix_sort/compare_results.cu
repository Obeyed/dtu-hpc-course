#include "radix_sort.h"
#include "utils.h"
#include "reference_calc.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  // change
  const size_t num_elems = 1 << 10;
  // define following as needed
  unsigned int* input_vals  = new unsigned int[num_elems];
  unsigned int* input_pos   = new unsigned int[num_elems];
  unsigned int* serial_output = new unsigned int[num_elems];
  unsigned int* output_pos  = new unsigned int[num_elems];

  // define initial values
  printf("INITIALISING RANDOM VALUES..\n");
  srand(time(NULL));
  for (unsigned int i = 0; i < num_elems; i++) {
    input_vals[i] = rand(); 
    input_pos[i] = i;
  }

  // call parallel code
  printf("CALLING GPU CODE..\n");
  unsigned int* parallel_output = radix_sort(input_vals, num_elems);

  // call serial code
  printf("CALLING REFERENCE..\n");
  reference_calculation(&input_vals[0],  &input_pos[0],
                        &serial_output[0], &output_pos[0],
                        num_elems);

  input_vals[0] = 10;
  // compare results
  printf("COMPARING RESULTS..\n");
  checkResultsExact(&serial_output[0], &parallel_output[0], num_elems);

  // if this is reached, then success
  printf("CORRECT!\n");

//  for (int i = 0; i < num_elems; i++)
//    printf("%u%s", result[i], ((i % 8 == 7) ? "\n" : "\t"));
//  printf("\n");

  return 0;
}
