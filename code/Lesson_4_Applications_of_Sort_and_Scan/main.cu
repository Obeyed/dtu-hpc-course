#include "radix_sort.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  const size_t numElems = 1 << 10;
  unsigned int* vals = new unsigned int[numElems];

  printf("INITIALISING RANDOM VALUES\n");

  srand(time(NULL));
  for (unsigned int i = 0; i < numElems; i++)
    vals[i] = rand(); 

  printf("CALLING RADIX SORT\n");

  unsigned int* result = radix_sort(vals, numElems);

  for (int i = 0; i < numElems; i++)
    printf("%u%s", result[i], ((i % 8 == 7) ? "\n" : "\t"));
  printf("\n");

  return 0;
}
