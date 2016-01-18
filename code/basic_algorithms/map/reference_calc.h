#ifndef REFERENCE_H__
#define REFERENCE_H__

void reference_calculation(unsigned int* const h_out,
                           unsigned int* const h_in, 
                           const size_t NUM_ELEMS) {
  for (unsigned int i = 0; i < NUM_ELEMS; i++) 
    h_out[i] = h_in[i];
}

#endif

