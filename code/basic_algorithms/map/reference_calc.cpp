#include "reference_calc.h"

void reference_calculation(unsigned int* const h_out,
                           unsigned int* const h_in, 
                           const unsigned int NUM_ELEMS) {
  for (unsigned int i = 0; i < NUM_ELEMS; i++) 
    h_out[i] = h_in[i];
}
