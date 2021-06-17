import logging
from typing import Optional

import numpy as np

from ._c import NUM_COST_TERMS

try:
    from numba import njit, prange
except ImportError:

    # In case numba is not install we create a dummy decorator
    def njit(*args, **kwargs):
        def _njit(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        if len(args) == 1 and callable(args[0]):
            return _njit(args[0])
        else:
            return _njit

    prange = range

logger = logging.getLogger(__name__)


@njit
def compute_dvdt_for_v(V, T, V_th):

    idx_o = 0
    for i, v in enumerate(V):
        if v > V_th:
            idx_o = i
            break

    if idx_o == 0:
        return np.inf

    return (V[idx_o] - V[idx_o - 1]) / (T[idx_o] - T[idx_o - 1])


# @njit
# def compute_APD_from_stim(V, T, t_stim, factor):
#     T_half = T.max() / 2
#     idx_T_half = np.argmin(np.abs(T - T_half))

#     # Set up threshold
#     V_max = np.max(V[:idx_T_half])
#     max_idx = np.argmax(V[:idx_T_half])
#     V_min = np.min(V)

#     th = V_min + (1 - factor / 100) * (V_max - V_min)

#     # % Find start time
#     t_start = t_stim
#     # % Find end time
#     t_end, idx2 = get_t_end(max_idx, V, T, th, t_end=T[-1])

#     return t_end - t_start


@njit
def get_t_start(max_idx, V, T, th, t_start=0):
    idx1 = 0
    for n in range(min(max_idx, len(T) - 1)):
        if V[n + 1] > th and V[n] <= th:  # n is lower point
            idx1 = n
            v_u = V[idx1]
            v_o = V[idx1 + 1]
            t_u = T[idx1]
            t_o = T[idx1 + 1]
            t_start = (
                (t_o - t_u) / (v_o - v_u) * (th - (t_o * v_u - t_u * v_o) / (t_o - t_u))
            )
            break
    return t_start, idx1


@njit
def get_t_end(max_idx, V, T, th, t_end=np.inf):
    idx2 = len(T)
    for n in range(max(1, max_idx), len(T)):
        if V[n - 1] > th and V[n] <= th:  # n is lower point
            idx2 = n
            v_u = V[idx2]
            v_o = V[idx2 - 1]
            t_u = T[idx2]
            t_o = T[idx2 - 1]
            t_end = (
                (t_o - t_u) / (v_o - v_u) * (th - (t_o * v_u - t_u * v_o) / (t_o - t_u))
            )
            break
    return t_end, idx2


@njit
def apd_up_xy(y: np.ndarray, t: np.ndarray, factor_x: int, factor_y: int) -> float:
    """Compute time from first intersection of
    APDx line to first intersection of APDy line
    """
    if factor_x > factor_y:
        # factor_x has to be larger than factor_y
        return -np.inf
    if factor_x == factor_y:
        return 0

    t_half = t.max() / 2
    idx_t_half = int(np.argmin(np.abs(t - t_half)))

    # Set up threshold
    y_max = np.max(y[:idx_t_half])
    max_idx = np.argmax(y[:idx_t_half])
    y_min = np.min(y)

    thx = y_min + (1 - factor_x / 100) * (y_max - y_min)
    tx, idx1 = get_t_start(max_idx, y, t, thx, t_start=0)
    thy = y_min + (1 - factor_y / 100) * (y_max - y_min)
    ty, idx1 = get_t_start(max_idx, y, t, thy, t_start=tx)
    return tx - ty


@njit
def compute_integral(V, T, factor):

    dt = T[1] - T[0]

    # Set up threshold
    V_max = np.max(V)
    max_idx = np.argmax(V)
    V_min = np.min(V)

    th = V_min + (1 - factor / 100) * (V_max - V_min)

    # Find start time
    dt_start = np.inf
    t_start, idx1 = get_t_start(max_idx, V, T, th)
    idx1 = idx1 + 1
    dt_start = T[idx1] - t_start

    # Find end time
    dt_end = np.inf
    t_end, idx2 = get_t_end(max_idx, V, T, th)
    idx2 = idx2 - 1
    dt_end = t_end - T[idx2]

    # Compute integral
    if np.isinf(dt_end) or np.isinf(dt_start):
        integral = np.inf
    else:
        integral = (
            dt * np.sum(V[idx1 + 1 : idx2] - th)
            + dt / 2 * (V[idx1] - th + V[idx2] - th)
            + dt_start / 2 * (V[idx1])
            + dt_end / 2 * (V[idx2])
        )

    return integral


@njit
def peak_and_repolarization(V, T, factor_low, factor_high):

    T_half = np.max(T) / 2
    idx_T_half = np.argmin(np.abs(T - T_half))

    # Set up threshold
    V_max = np.max(V[1:idx_T_half])
    max_idx = np.argmax(V[1:idx_T_half])
    V_min = np.min(V)

    th_high = V_min + (1 - factor_high / 100) * (V_max - V_min)
    th_low = V_min + (1 - factor_low / 100) * (V_max - V_min)

    # Find start time upstroke
    t_start_up, idx1 = get_t_start(max_idx, V, T, th_high, t_start=0)

    # Find end time upstroke
    t_end_up, idx1 = get_t_start(max_idx, V, T, th_low, t_start=T[max_idx])

    # Find start time repolarization
    t_start_down, idx2 = get_t_end(max_idx, V, T, th_low, t_end=t_end_up)
    # Find end time repolarization
    t_end_down, idx2 = get_t_end(max_idx, V, T, th_high, t_end=t_end_up)

    time_up = t_end_up - t_start_up
    time_down = t_end_down - t_start_down
    return time_up, time_down


@njit
def compute_dvdt_max(V, T):
    return np.max(np.divide(V[1:] - V[:-1], T[1:] - T[:-1]))


@njit
def apd(V, T, factor):

    T_half = T.max() / 2
    idx_T_half = np.argmin(np.abs(T - T_half))

    # Set up threshold
    V_max = np.max(V[:idx_T_half])
    max_idx = np.argmax(V[:idx_T_half])
    V_min = np.min(V)

    th = V_min + (1 - factor / 100) * (V_max - V_min)
    t_start, idx1 = get_t_start(max_idx, V, T, th, t_start=0)
    t_end, idx2 = get_t_end(max_idx, V, T, th, t_end=np.inf)

    return t_end - t_start


@njit
def cost_terms_trace(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.zeros(NUM_COST_TERMS // 2)
    return _cost_terms_trace(y, t, R)


@njit
def _cost_terms_trace(v, t, R):
    # R = np.zeros(24)
    R[:] = np.inf
    # Max and min membrane potential
    R[0] = np.max(v)
    R[1] = np.min(v)
    R[2] = t[np.argmax(v)]

    # Upstroke velocity
    R[3] = compute_dvdt_max(v, t)

    i = 4
    for factor in np.arange(10, 95, 5):
        R[i] = apd(v, t, factor)
        i += 1
    for x in np.arange(20, 61, 20):
        for y in np.arange(x + 20, 81, 20):
            R[i] = apd_up_xy(v, t, x, y)
            i += 1

    R[27] = compute_integral(v, t, 30)
    R[28], R[29] = peak_and_repolarization(v, t, 20, 80)

    return R


@njit
def _cost_terms(v, ca, t_v, t_ca, R):
    _cost_terms_trace(v, t_v, R[: NUM_COST_TERMS // 2])
    _cost_terms_trace(ca, t_ca, R[NUM_COST_TERMS // 2 :])
    return R


@njit
def cost_terms(
    v: np.ndarray,
    ca: np.ndarray,
    t_v: np.ndarray,
    t_ca: np.ndarray,
) -> np.ndarray:
    R = np.zeros(NUM_COST_TERMS, dtype=np.float64)
    return _cost_terms(v, ca, t_v, t_ca, R)


def all_cost_terms(
    arr: np.ndarray,
    t: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_w: Optional[int] = NUM_COST_TERMS,
) -> np.ndarray:
    if mask is None:
        num_parameter_sets = arr.shape[-1]
        mask = np.zeros(num_parameter_sets)
    return _all_cost_terms(arr, t, mask, num_w)


@njit(parallel=True)
def _all_cost_terms(arr_, t, mask, num_w):
    arr = transpose_trace_array(arr_)
    num_parameter_sets = arr.shape[0]
    cost = np.zeros((num_parameter_sets, num_w), dtype=np.float64)
    for i in prange(num_parameter_sets):
        if mask[i]:
            cost[i, :] = np.inf
            continue

        cost[i, :] = cost_terms(v=arr[i, 0, :], ca=arr[i, 1, :], t_v=t, t_ca=t)
    return cost


@njit(parallel=True)
def transpose_trace_array(arr):
    old_shape = arr.shape
    num_trace_points, num_traced_states, num_parameter_sets = old_shape
    new_shape = num_parameter_sets, num_traced_states, num_trace_points

    new_arr = np.empty(new_shape, dtype=arr.dtype)
    for p in prange(num_parameter_sets):
        for s in range(num_traced_states):
            for t in range(num_trace_points):
                new_arr[p, s, t] = arr[t, s, p]
    return new_arr
