#include "reference_calc.h"

void reference_calculation(unsigned int* const h_out, 
                           unsigned int* const h_in, 
                           const unsigned int NUM_ELEMS) {
  h_out[0] = h_in[0];
  for (unsigned int l = 1; l < IN_SIZE; ++l)
    h_out[l] = h_out[l-1] + h_in[l];
}
