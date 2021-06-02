#ifndef COST_TERMS_H
#define COST_TERMS_H

#include <stdint.h>

#include "common.h"

// new Python API
typedef int (*progress_update_func_ptr)(int);

#define NUM_COST_TERMS 60

void CDECL all_cost_terms(double *R, double *traces, double *t, uint8_t *mask, long length,
                          long num_parameter_sets, progress_update_func_ptr progress_update);

#endif
