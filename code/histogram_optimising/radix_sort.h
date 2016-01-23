#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "gputimer.h"

unsigned int** radix_sort(float*&, unsigned int**, const size_t, const size_t);

#endif
