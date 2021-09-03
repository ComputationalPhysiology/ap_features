#include "beeler_reuter_1977_version06.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// Gotran generated C/C++ code for the "base_model" model

void ode_solve_forward_euler(double *u, const double *parameters, double *u_values,
                             double *t_values, int num_timesteps, double dt)
{
    double t;
    int save_it = 1;
    int it, j;
    for (it = 1; it <= num_timesteps; it++) {
        t = t_values[it - 1];
        //forward_explicit_euler(u, t, dt, parameters, celltype);
        forward_explicit_euler(u, t, dt, parameters);
        for (j = 0; j < NUM_STATES; j++) {
            u_values[save_it * NUM_STATES + j] = u[j];
        }
        save_it++;
    }
}

void ode_solve_rush_larsen(double *u, const double *parameters, double *u_values, double *t_values,
                           int num_timesteps, double dt)
{
    double t;
    int save_it = 1;
    int it, j;
    for (it = 1; it <= num_timesteps; it++) {
        t = t_values[it - 1];
        forward_rush_larsen(u, t, dt, parameters);
        for (j = 0; j < NUM_STATES; j++) {
            u_values[save_it * NUM_STATES + j] = u[j];
        }
        save_it++;
    }
}

void monitored_values(double *monitored, double *states, double *parameters, double *u, double *m,
                      double *t_values, int length)
{
    double t;
    int i, j;
    for (i = 0; i <= length; i++) {
        t = t_values[i];
        for (j = 0; j < NUM_STATES; j++) {
            u[j] = states[i * NUM_STATES + j];
        }
        monitor(u, t, parameters, m);
        for (j = 0; j < NUM_MONITORED; j++) {
            monitored[i * NUM_MONITORED + j] = m[j];
        }
    }
}

void monitored_values_single(double *monitored, double *states, double *parameters, double *u,
                             double *m, double *t_values, int length, int index)
{
    double t;
    int i, j;
    for (i = 0; i <= length; i++) {
        t = t_values[i];
        for (j = 0; j < NUM_STATES; j++) {
            u[j] = states[i * NUM_STATES + j];
        }
        monitor(u, t, parameters, m);
        monitored[i] = m[index];
    }
}

int state_count()
{
    return NUM_STATES;
}

int parameter_count()
{
    return NUM_PARAMS;
}

int monitor_count()
{
    return NUM_MONITORED;
}

int main(int argc, char *argv[])
{
    double t_start = 0;
    double dt = 0.02E-3;
    int num_timesteps = (int) 1000000;
    if (argc > 1) {
        num_timesteps = atoi(argv[1]);
        printf("num_timesteps set to %d\n", num_timesteps);
        if (num_timesteps <= 0) {
            exit(EXIT_FAILURE);
        }
    }

    unsigned int num_states = NUM_STATES;
    size_t states_size = num_states * sizeof(double);

    unsigned int num_parameters = NUM_PARAMS;
    size_t parameters_size = num_parameters * sizeof(double);

    double *states = malloc(states_size);
    double *parameters = malloc(parameters_size);
    init_parameters_values(parameters);

    double t = t_start;

    struct timespec timestamp_start, timestamp_now;
    double time_elapsed;

    // forward euler
    printf("Scheme: Forward Euler\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_start);
    init_state_values(states);
    int it;
    for (it = 0; it < num_timesteps; it++) {
        forward_explicit_euler(states, t, dt, parameters);
        t += dt;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_now);
    time_elapsed = timestamp_now.tv_sec - timestamp_start.tv_sec
                   + 1E-9 * (timestamp_now.tv_nsec - timestamp_start.tv_nsec);
    printf("Computed %d time steps in %g s. Time steps per second: %g\n", num_timesteps,
           time_elapsed, num_timesteps / time_elapsed);
    printf("\n");

    // Rush Larsen
    printf("Scheme: Rush Larsen (exp integrator on all gates)\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_start);
    init_state_values(states);
    for (it = 0; it < num_timesteps; it++) {
        forward_rush_larsen(states, t, dt, parameters);
        t += dt;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_now);
    time_elapsed = timestamp_now.tv_sec - timestamp_start.tv_sec
                   + 1E-9 * (timestamp_now.tv_nsec - timestamp_start.tv_nsec);
    printf("Computed %d time steps in %g s. Time steps per second: %g\n", num_timesteps,
           time_elapsed, num_timesteps / time_elapsed);
    printf("\n");

    free(states);
    free(parameters);

    return 0;
}
