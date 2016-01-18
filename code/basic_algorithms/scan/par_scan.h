#ifndef PAR_SCAN_H
#define PAR_SCAN_H

#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

void par_scan(unsigned int* const, unsigned int* const, const size_t);

#endif
