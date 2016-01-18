#ifndef PAR_MAP_H
#define PAR_MAP_H

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

void par_map(unsigned int* const, unsigned int* const, const size_t);

#endif
