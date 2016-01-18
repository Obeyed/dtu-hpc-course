#include "reference_calc.h"

void reference_calculation(unsigned int* const h_out, 
                           unsigned int* const h_in, 
                           const unsigned int NUM_ELEMS,
                           const unsigned int BIN_SIZE) {
  for (unsigned int l = 0; l < NUM_ELEMS; ++l)
    h_out[(h_in[l] % BIN_SIZE)]++;
}
