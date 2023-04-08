#ifndef COST_TERMS_H
#define COST_TERMS_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif


#include "common.h"

// new Python API
typedef int (*progress_update_func_ptr)(int);

#define NUM_COST_TERMS 60

void CDECL all_cost_terms(double *R, double *traces, double *t, uint8_t *mask, long length,
                          long num_parameter_sets, progress_update_func_ptr progress_update);

int CDECL get_num_cost_terms();
double CDECL apd(double *V, double *t, int factor, int length, double *T_diff_buf);
double CDECL apd_up_xy(double *V, double *t, int factor_x, int factor_y, int length,
                       double *T_diff_buf);
#endif
