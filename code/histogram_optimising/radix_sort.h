#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

unsigned int** radix_sort(unsigned int**, const size_t, const size_t);

#endif
