#ifndef PAR_REDUCE_H
#define PAR_REDUCE_H

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

void par_reduce(unsigned int* const, unsigned int* const, const size_t);

#endif
