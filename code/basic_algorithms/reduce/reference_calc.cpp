#include "reference_calc.h"

void reference_calculation(unsigned int* const h_out,
                            unsigned int* const h_in,
                            const unsigned int NUM_ELEMS) {
  h_out[0] = 0;
  for (unsigned int l = 0; l < NUM_ELEMS; ++l)
    h_out[0] += h_in[l];
}
