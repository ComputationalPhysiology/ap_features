import ctypes
import logging
import os
import time
from ctypes import c_double, c_int, c_long, c_uint8, c_void_p
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np
import tqdm

try:
    from numba import njit, prange
except ImportError:

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

HERE = Path(__file__).absolute().parent


def py_update_progress(progress_bar=None):
    if progress_bar is None:
        progress_bar = mock.Mock()
        progress_bar.n = 0

    @ctypes.CFUNCTYPE(c_int, c_int)
    def py_update_progress_wrap(current_step):
        increment = current_step - progress_bar.n
        progress_bar.update(increment)
        return 0

    return py_update_progress_wrap


def load_library(name: str) -> ctypes.CDLL:

    try:
        libname = next(
            f.name for f in HERE.iterdir() if f.name.startswith("cost_terms")
        )
    except StopIteration:
        raise FileNotFoundError(f"Could not find shared library for {name}")

    lib = np.ctypeslib.load_library(libname, HERE)

    lib_path = lib._name
    lib_mtime_float = os.path.getmtime(lib_path)
    lib_mtime_struct = time.localtime(lib_mtime_float)
    lib_mtime_str = time.asctime(lib_mtime_struct)
    logger.debug(f"Loading library '{lib_path}' last modified {lib_mtime_str}")
    return lib


lib = load_library("cost_terms")

uint8_array = np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1, flags="contiguous")
float64_array = np.ctypeslib.ndpointer(dtype=c_double, ndim=1, flags="contiguous")
float64_array_2d = np.ctypeslib.ndpointer(dtype=c_double, ndim=2, flags="contiguous")
float64_array_3d = np.ctypeslib.ndpointer(dtype=c_double, ndim=3, flags="contiguous")

lib.apd.argtypes = [float64_array, float64_array, c_int, c_int, float64_array]
lib.apd.restype = c_double

lib.apd_up_xy.argtypes = [
    float64_array,
    float64_array,
    c_int,
    c_int,
    c_int,
    float64_array,
]
lib.apd_up_xy.restype = c_double

lib.cost_terms_trace.argtypes = [float64_array, float64_array, float64_array, c_int]
lib.cost_terms_trace.restype = None

lib.get_num_cost_terms.argtypes = []
lib.get_num_cost_terms.restype = c_int

lib.full_cost_terms.argtypes = [
    float64_array,
    float64_array,
    float64_array,
    float64_array,
    c_int,
]
lib.full_cost_terms.restype = None

lib.all_cost_terms.argtypes = [
    float64_array_2d,
    float64_array_3d,
    float64_array,
    uint8_array,
    c_long,
    c_long,
    c_void_p,
]
lib.all_cost_terms.restype = None

NUM_COST_TERMS = lib.get_num_cost_terms()


def apd_c(V, T, factor):
    return lib.apd(V[...], T[...], factor, T.size, V.copy())


def apd_up_xy_c(V, T, factor_x, factor_y):
    return lib.apd_up_xy(V[...], T[...], factor_x, factor_y, T.size, V.copy())


def cost_terms_trace_c(V, T):

    R = np.zeros(NUM_COST_TERMS // 2)
    lib.cost_terms_trace(R, V[...], T[...], T.size)
    return R


def cost_terms_c(v, ca, t_v, t_ca):
    R = np.zeros(NUM_COST_TERMS)
    lib.cost_terms_trace(R[: NUM_COST_TERMS // 2], v[...], t_v[...], t_v.size)
    lib.cost_terms_trace(R[NUM_COST_TERMS // 2 :], ca[...], t_ca[...], t_ca.size)
    return R


def all_cost_terms_c(
    arr: np.ndarray,
    t: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize_time: bool = True,
) -> np.ndarray:
    traces = transpose_trace_array(arr[...])
    num_sets = traces.shape[0]
    if mask is None:
        mask = np.zeros(num_sets, dtype=np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    verbose = logger.level <= 20

    if verbose:
        progress_bar = tqdm.tqdm(total=num_sets)
        progress_bar.set_description("Computing cost terms")
    else:
        progress_bar = None

    update_progress_func = py_update_progress(progress_bar)

    if normalize_time:
        t = t - t[0]
    R = np.zeros((num_sets, NUM_COST_TERMS))
    lib.all_cost_terms(
        R, traces, t[...], mask[...], t.size, num_sets, update_progress_func
    )
    return R


def list_cost_function_terms_trace(key=""):

    apd_key = "APD"
    if key.lower() == "ca":
        apd_key = "CaD"

    if key != "":
        key += "_"

    lst = (
        [f"{key}max", f"{key}min", f"{key}t_max", f"d{key}dt_max"]
        + [f"{apd_key}{apd}" for apd in np.arange(10, 95, 5, dtype=int)]
        + [
            f"{apd_key}_up_{x}{y}"
            for x in np.arange(20, 61, 20, dtype=int)
            for y in np.arange(x + 20, 81, 20, dtype=int)
        ]
        + [f"{key}int_30", f"{key}t_up", f"{key}t_down"]
    )
    return lst


def list_cost_function_terms():
    return list_cost_function_terms_trace("V") + list_cost_function_terms_trace("Ca")


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


@njit
def compute_APD_from_stim(V, T, t_stim, factor):
    T_half = T.max() / 2
    idx_T_half = np.argmin(np.abs(T - T_half))

    # Set up threshold
    V_max = np.max(V[:idx_T_half])
    max_idx = np.argmax(V[:idx_T_half])
    V_min = np.min(V)

    th = V_min + (1 - factor / 100) * (V_max - V_min)

    # % Find start time
    t_start = t_stim
    # % Find end time
    t_end, idx2 = get_t_end(max_idx, V, T, th, t_end=T[-1])

    return t_end - t_start


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
def compute_APDUpxy(V, T, x=20, y=80):
    """Compute time from first intersection of
    APDx line to first intersection of APDy line
    """
    if x > y:
        # x has to be larger thay y
        return -np.inf
    if x == y:
        return 0

    T_half = T.max() / 2
    idx_T_half = np.argmin(np.abs(T - T_half))

    # Set up threshold
    V_max = np.max(V[:idx_T_half])
    max_idx = np.argmax(V[:idx_T_half])
    V_min = np.min(V)

    thx = V_min + (1 - x / 100) * (V_max - V_min)
    tx, idx1 = get_t_start(max_idx, V, T, thx, t_start=0)
    thy = V_min + (1 - y / 100) * (V_max - V_min)
    ty, idx1 = get_t_start(max_idx, V, T, thy, t_start=tx)
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
    # try:
    t_end, idx2 = get_t_end(max_idx, V, T, th)
    idx2 = idx2 - 1
    dt_end = t_end - T[idx2]
    # except Exception as ex:
    # print(ex)
    # print('Fail at {}'.format(idx2))

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
def compute_APD(V, T, factor):

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
def cost_terms_trace(v, t):
    R = np.zeros(NUM_COST_TERMS // 2)
    return _cost_terms_trace(v, t, R)


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
    for apd in np.arange(10, 95, 5):
        R[i] = compute_APD(v, t, apd)
        i += 1
    for x in np.arange(20, 61, 20):
        for y in np.arange(x + 20, 81, 20):
            R[i] = compute_APDUpxy(v, t, x, y)
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
def cost_terms(v, ca, t_v, t_ca):
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
