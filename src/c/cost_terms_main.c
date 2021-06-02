#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cost_terms.h"

/* Minimal sample program to run standalone with valgrind to check for memory errors */

int progress_updater(int i)
{
    printf("Progress: %7d\n", i);
    return 0;
}

int main()
{
    long num_parameter_sets = 250000;
    int trace_length = 200;
    int num_traced_states = 2;
    progress_update_func_ptr progress_update = 0;

    double *R = malloc(num_parameter_sets * NUM_COST_TERMS * sizeof(double));
    double *traces = malloc(num_parameter_sets * num_traced_states * trace_length * sizeof(double));
    double *t = malloc(trace_length * sizeof(double));
    uint8_t *mask = malloc(num_parameter_sets * sizeof(uint8_t));
    memset(mask, 0, num_parameter_sets * sizeof(uint8_t));

    // initialise arrays
    for (int i = 0; i < trace_length; i++) {
        t[i] = i;
    }

    for (long i = 0; i < num_parameter_sets; i++) {
        // voltage
        for (int j = 0; j < trace_length; j++) {
            long ind = (i * num_traced_states) * trace_length + j;
            traces[ind] = 0.01 * j;
        }
        // calcium
        for (int j = 0; j < trace_length; j++) {
            long ind = (i * num_traced_states + 1) * trace_length + j;
            traces[ind] = 0.01 * j;
        }
    }

    struct timespec pre, post;

#ifdef _OS_WINDOWS
    timespec_get(&pre, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC_RAW, &pre);
#endif

    // compute cost terms
    //all_cost_terms(R, traces, t, mask, trace_length, num_parameter_sets, NULL);
    all_cost_terms(R, traces, t, mask, trace_length, num_parameter_sets, &progress_updater);
#ifdef _OS_WINDOWS
    timespec_get(&post, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC_RAW, &post);
#endif

    double time_elapsed = post.tv_sec - pre.tv_sec + 1E-9 * (post.tv_nsec - pre.tv_nsec);

    printf("Time elapsed computing cost terms: %g s\n", time_elapsed);

    free(R);
    free(traces);
    free(t);
    free(mask);
}
