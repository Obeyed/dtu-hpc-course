#include "radix_sort.h"

#include <stdlib.h>
#include <time.h>

int main(void) {
  const size_t numElems = 16;
  unsigned int* vals = new unsigned int[numElems];

  srand(time(NULL));
  for (unsigned int i = 0; i < numElems; i++)
    vals[i] = rand(); 

  printf("CALLING RADIX SORT");

  unsigned int* result = radix_sort(vals, numElems);

  return 0;
}
