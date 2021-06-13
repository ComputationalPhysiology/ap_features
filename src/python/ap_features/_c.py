import ctypes
import logging
from ctypes import c_int
from typing import Callable, Optional
from unittest import mock

import numpy as np
import tqdm
from scipy import interpolate

from .lib import lib
from .utils import Array, _check_factor

logger = logging.getLogger(__name__)
NUM_COST_TERMS = lib.get_num_cost_terms()


def py_update_progress(progress_bar: Optional[tqdm.tqdm] = None) -> Callable:
    """Helper function to be passed to C that
    will update the progressbar

    Parameters
    ----------
    progress_bar : Optional[tqdm.tqdm], optional
        A tqdm progress bar, by default None

    Returns
    -------
    Callable
        The function to be passed to C
    """
    if progress_bar is None:
        progress_bar = mock.Mock()
        progress_bar.n = 0

    @ctypes.CFUNCTYPE(c_int, c_int)
    def py_update_progress_wrap(current_step):
        increment = current_step - progress_bar.n
        progress_bar.update(increment)
        return 0

    return py_update_progress_wrap


def apd(y: Array, t: Array, factor: interpolate) -> float:
    """Return the action potential duration at the given factor.

    Parameters
    ----------
    y : Array
        The signal
    t : Array
        The time stamps
    factor : int
        The factor value between 0 and 100

    Returns
    -------
    float
        The action potential duration

    """
    _check_factor(factor)
    return lib.apd(
        np.array(y)[...], np.array(y)[...], int(factor), len(y), np.array(y).copy()
    )


def apd_up_xy_c(y: Array, t: Array, factor_x: int, factor_y: int) -> float:
    return lib.apd_up_xy(
        np.array(y)[...],
        np.array(t)[...],
        factor_x,
        factor_y,
        len(t),
        np.array(y).copy(),
    )


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
    # check that the number of trace points is consistent
    assert t.shape[0] == arr.shape[0]
    num_trace_points = t.shape[0]

    from ._numba import transpose_trace_array

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
        R, traces, t[...], mask[...], num_trace_points, num_sets, update_progress_func
    )
    if progress_bar:
        progress_bar.close()
    return R
