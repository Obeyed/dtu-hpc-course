#ifndef PAR_HISTOGRAM_H
#define PAR_HISTOGRAM_H

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

void par_histogram(unsigned int* const, unsigned int* const, const size_t, const unsigned int);

#endif
