#ifndef REFERENCE_H__
#define REFERENCE_H__

unsigned int* ser_map(unsigned int* h_in, const size_t NUM_ELEMS) {
  unsigned int* h_out = new unsigned int[NUM_ELEMS];
  for (unsigned int i = 0; i < NUM_ELEMS; i++) 
    h_out[i] = h_in[i];

  return h_out;
}

#endif

