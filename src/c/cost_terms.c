#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "cost_terms.h"

#define PROGRESS_UPDATE_PERIOD                                                                     \
    (1 << 10) // number of parameter sets to compute per thread between each progress update


int CDECL get_num_cost_terms()
{
    return NUM_COST_TERMS;
}

double CDECL array_max_double(double *x, int length)
{
    double m = x[0];
    int i;
    for (i = 1; i < length; i++) {
        if (x[i] > m) {
            m = x[i];
        }
    }
    return m;
}

int CDECL max_int(int x1, int x2)
{
    if (x1 > x2) {
        return x1;
    } else {
        return x2;
    }
}

int CDECL min_int(int x1, int x2)
{
    if (x1 < x2) {
        return x1;
    } else {
        return x2;
    }
}

double CDECL array_min_double(double *x, int length)
{
    double m = x[0];
    int i;
    for (i = 1; i < length; i++) {
        if (x[i] < m) {
            m = x[i];
        }
    }
    return m;
}

int CDECL argmin(double *x, int length)
{
    double m = x[0];
    int min_idx = 0;
    int i;
    for (i = 1; i < length; i++) {
        if (x[i] < m) {
            m = x[i];
            min_idx = i;
        }
    }
    return min_idx;
}

int CDECL argmax(double *x, int length)
{
    double m = x[0];
    int max_idx = 0;
    int i;
    for (i = 1; i < length; i++) {
        if (x[i] > m) {
            m = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void CDECL sub_abs(double *s, double *x, double y, int length)
{
    int i;
    for (i = 0; i < length; i++) {
        s[i] = fabs(x[i] - y);
    }
}

int CDECL get_dt_start(int max_idx, double *V, double *T, double th, int length, double dt_start)
{
    int idx = 0;
    int N = min_int(length - 1, max_idx);
    double v_u, v_o, t_u, t_o, t_start;

    for (int n = 0; n < N; n++) {
        if (V[n + 1] > th && V[n] <= th) {
            idx = n;
            v_u = V[idx];
            v_o = V[idx + 1];
            t_u = T[idx];
            t_o = T[idx + 1];
            t_start = ((t_o - t_u) / (v_o - v_u) * (th - (t_o * v_u - t_u * v_o) / (t_o - t_u)));
            idx = idx + 1;
            dt_start = T[idx] - t_start;
            break;
        }
    }
    return idx;
}

double CDECL get_t_start(int max_idx, double *V, double *T, double th, int length, double t_start)
{
    int idx = 0;
    double v_u, v_o, t_u, t_o;
    int N = min_int(length - 1, max_idx);

    for (int n = 0; n < N; n++) {
        if (V[n + 1] > th && V[n] <= th) {
            idx = n;
            v_u = V[idx];
            v_o = V[idx + 1];
            t_u = T[idx];
            t_o = T[idx + 1];
            t_start = ((t_o - t_u) / (v_o - v_u) * (th - (t_o * v_u - t_u * v_o) / (t_o - t_u)));
            break;
        }
    }
    return t_start;
}

int CDECL get_dt_end(int max_idx, double *V, double *T, double th, int length, double dt_end)
{
    int idx = 0;
    double v_u, v_o, t_u, t_o, t_end;
    int N = max_int(1, max_idx);
    int n;

    for (n = N; n < length; n++) {
        if (V[n - 1] > th && V[n] <= th) {
            idx = n;
            v_u = V[idx];
            v_o = V[idx - 1];
            t_u = T[idx];
            t_o = T[idx - 1];
            t_end = ((t_o - t_u) / (v_o - v_u) * (th - (t_o * v_u - t_u * v_o) / (t_o - t_u)));
            idx = idx - 1;
            dt_end = t_end - T[idx];
            break;
        }
    }
    return idx;
}

double CDECL get_t_end(int max_idx, double *V, double *T, double th, int length, double t_end)
{
    int idx = 0;
    int N = max_int(1, max_idx);
    double v_u, v_o, t_u, t_o;
    int n;

    for (n = N; n < length; n++) {
        if (V[n - 1] > th && V[n] <= th) {
            idx = n;
            v_u = V[idx];
            v_o = V[idx - 1];
            t_u = T[idx];
            t_o = T[idx - 1];
            t_end = ((t_o - t_u) / (v_o - v_u) * (th - (t_o * v_u - t_u * v_o) / (t_o - t_u)));
            break;
        }
    }
    return t_end;
}

double CDECL dv_dt_max(double *V, double *t, int length)
{

    double max_dv_dt = 0.0;
    double dv_dt;
    int i;
    for (i = 0; i < length - 1; i++) {
        dv_dt = (V[i + 1] - V[i]) / (t[i + 1] - t[i]);
        if (dv_dt > max_dv_dt) {
            max_dv_dt = dv_dt;
        }
    }
    return max_dv_dt;
}

double CDECL apd(double *V, double *t, int factor, int length, double *T_diff_buf)
{

    double T_max = array_max_double(t, length);
    double T_half = T_max / 2;
    sub_abs(T_diff_buf, t, T_half, length);
    int idx_T_half = argmin(T_diff_buf, length);
    double V_max = array_max_double(V, idx_T_half);
    int max_idx = argmax(V, idx_T_half);
    double V_min = array_min_double(V, length);
    double th = V_min + (1 - factor / 100.0) * (V_max - V_min);
    double t_start = get_t_start(max_idx, V, t, th, length, 0.0);
    double t_end = get_t_end(max_idx, V, t, th, length, INFINITY);

    return t_end - t_start;
}

double CDECL apd_up_xy(double *V, double *t, int factor_x, int factor_y, int length,
                       double *T_diff_buf)
{
    if (factor_x > factor_y) {
        return -INFINITY;
    }
    if (factor_x == factor_y) {
        return 0.0;
    }

    double T_max = array_max_double(t, length);
    double T_half = T_max / 2;
    sub_abs(T_diff_buf, t, T_half, length);
    int idx_T_half = argmin(T_diff_buf, length);
    double V_max = array_max_double(V, idx_T_half);
    int max_idx = argmax(V, idx_T_half);
    double V_min = array_min_double(V, length);

    double thx = V_min + (1 - factor_x / 100.0) * (V_max - V_min);
    double tx = get_t_start(max_idx, V, t, thx, length, 0.0);

    double thy = V_min + (1 - factor_y / 100.0) * (V_max - V_min);
    double ty = get_t_start(max_idx, V, t, thy, length, tx);

    return tx - ty;
}

double CDECL time_up(double *V, double *t, int length, int factor_low, int factor_high,
                     double *T_diff_buf)
{

    double T_max = array_max_double(t, length);
    double T_half = 4 * T_max / 5;

    sub_abs(T_diff_buf, t, T_half, length);
    int idx_T_half = argmin(T_diff_buf, length);

    double V_max = array_max_double(V, idx_T_half);

    int max_idx = argmax(V, idx_T_half);

    double V_min = array_min_double(V, length);

    double th_low = V_min + (1 - factor_low / 100.0) * (V_max - V_min);
    double th_high = V_min + (1 - factor_high / 100.0) * (V_max - V_min);


    double t_start_up = get_t_start(max_idx, V, t, th_high, length, 0.0);
    double t_end_up = get_t_start(max_idx, V, t, th_low, length, t[max_idx]);
    return t_end_up - t_start_up;
}

double CDECL time_down(double *V, double *t, int length, int factor_low, int factor_high,
                       double *T_diff_buf)
{

    double T_max = array_max_double(t, length);
    double T_half = 4 * T_max / 5;

    sub_abs(T_diff_buf, t, T_half, length);
    int idx_T_half = argmin(T_diff_buf, length);

    double V_max = array_max_double(V, idx_T_half);

    int max_idx = argmax(V, idx_T_half);

    double V_min = array_min_double(V, length);

    double th_low = V_min + (1 - factor_low / 100.0) * (V_max - V_min);
    double th_high = V_min + (1 - factor_high / 100.0) * (V_max - V_min);

    double t_end_up = get_t_start(max_idx, V, t, th_low, length, t[max_idx]);
    double t_start_down = get_t_end(max_idx, V, t, th_low, length, t_end_up);
    double t_end_down = get_t_end(max_idx, V, t, th_high, length, t_end_up);
    return t_end_down - t_start_down;
}

double CDECL trapz(double *V, double *T, int length, double extra)
{
    double sum = 0.0;
    double h;

    // Compute the integral of sin(X) using Trapezoidal numerical integration method
    for (int j = 0; j < length - 1; ++j) {
        h = T[j + 1] - T[j];
        // if (j == 0 || j == length - 1) // for the first and last elements
        //     sum += h * (V[j] + extra) / 2;
        // else
        sum += h * (V[j] + V[j + 1] + 2 * extra) / 2; // the rest of data
    }
    return sum;
}

double CDECL compute_integral(double *V, double *t, int length, int factor)
{

    double V_max = array_max_double(V, length);

    int max_idx = argmax(V, length);

    double V_min = array_min_double(V, length);

    double th = V_min + (1 - factor / 100.0) * (V_max - V_min);

    int idx1 = get_dt_start(max_idx, V, t, th, length, INFINITY);

    int idx2 = get_dt_end(max_idx, V, t, th, length, INFINITY);

    int n = idx2 - idx1 - 1;
    if (idx2 < idx1 + 3) {
        return INFINITY;
    }


    return trapz(V + idx1, t + idx1, n, -th);
}

void CDECL cost_terms_trace(double *R, double *V, double *t, int length)
{
    int factor = 10;
    size_t V_size = length * sizeof(double);
    double *T_diff_buf = malloc(V_size);
    R[0] = array_max_double(V, length);
    R[1] = array_min_double(V, length);
    R[2] = argmax(V, length);
    R[3] = dv_dt_max(V, t, length); // dV/dt
    int i;
    for (i = 4; i < 21; i++) {
        R[i] = apd(V, t, factor, length, T_diff_buf);
        factor += 5;
    }
    for (int factor_x = 20; factor_x <= 60; factor_x += 20) {
        for (int factor_y = factor_x + 20; factor_y <= 80; factor_y += 20) {
            R[i] = apd_up_xy(V, t, factor_x, factor_y, length, T_diff_buf);
            i++;
        }
    }

    R[27] = compute_integral(V, t, length, 30);          // int30
    R[28] = time_up(V, t, length, 20, 80, T_diff_buf);   // t_up
    R[29] = time_down(V, t, length, 20, 80, T_diff_buf); // t_down
    free(T_diff_buf);
}

void CDECL full_cost_terms(double *R, double *V, double *Ca, double *t, int length)
{
    cost_terms_trace(R, V, t, length);
    cost_terms_trace(R + 30, Ca, t, length);
}

void CDECL fill_cost_array_with_inf(double *R)
{
    // printf("fill cost array with inf");
    for (int i = 0; i < NUM_COST_TERMS; i++) {
        R[i] = INFINITY;
    }
    // printf("done fill cost array with inf");
}


void CDECL all_cost_terms(double *R, double *traces, double *t, uint8_t *mask, long length,
                          long num_parameter_sets, progress_update_func_ptr progress_update)
{
#if defined(_OPENMP)
    const int num_threads = omp_get_max_threads();
#else
    const int num_threads = 1;
#endif
    int stride = 8; // we choose stride to be CACHE_LINE_SIZE / sizeof(long) = 8
    long *parameter_sets_computed_per_thread = calloc(num_threads * stride, sizeof(long));
    /* we pad the entries in parameter_sets_computed_per_thread so that each thread works on a separate 64 byte cache line */

#pragma omp parallel
    {
#if defined(_OPENMP)
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif
        // declare loop variable outside loop to make the Microsoft compiler happy
        long n;
#pragma omp for
        for (n = 0; n < num_parameter_sets; n++) {
            if (mask[n] == 1) {
                fill_cost_array_with_inf(R + NUM_COST_TERMS * n);
            } else {
                full_cost_terms((R + NUM_COST_TERMS * n), traces + (2 * n * length),
                                traces + ((2 * n + 1) * length), t, length);
            }

            // increase the number of parameter sets computed by this thread
            parameter_sets_computed_per_thread[thread_id * stride]++;
            if (thread_id == 0 && progress_update && n > 0 && n % PROGRESS_UPDATE_PERIOD == 0) {
                // let the thread 0 sum up the contributions for all threads
                long parameter_sets_computed = 0;
                for (int t = 0; t < num_threads; t++) {
                    parameter_sets_computed += parameter_sets_computed_per_thread[t * stride];
                }

                /* note that the callback function takes an int parameter,
                so we will have overflow above ~2 billion parameter sets */
                progress_update(parameter_sets_computed);
            }
        }
    }
    // perform a final unconditional progress update
    // if the number of parameter sets is low, this will be the only progress update.
    progress_update(num_parameter_sets);
    free(parameter_sets_computed_per_thread);
}

#ifdef _OS_WINDOWS
void PyInit_libcost_terms()
{
}
#endif
