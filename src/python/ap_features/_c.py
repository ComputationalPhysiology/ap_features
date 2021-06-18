import ctypes
import logging
from ctypes import c_int
from typing import Callable
from typing import Optional
from unittest import mock

import numpy as np
import tqdm
from scipy import interpolate

from .lib import lib
from .utils import _check_factor
from .utils import numpyfy

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


def apd(y: np.ndarray, t: np.ndarray, factor: interpolate) -> float:
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
    y = to_c_contigous(y)
    t = to_c_contigous(t)
    return lib.apd(
        np.array(y)[...],
        np.array(t)[...],
        int(factor),
        len(y),
        np.array(y).copy(),
    )


def apd_up_xy(y: np.ndarray, t: np.ndarray, factor_x: int, factor_y: int) -> float:
    y = to_c_contigous(y)
    t = to_c_contigous(t)

    return lib.apd_up_xy(
        np.array(y)[...],
        np.array(t)[...],
        factor_x,
        factor_y,
        len(t),
        np.array(y).copy(),
    )


def cost_terms_trace(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.zeros(NUM_COST_TERMS // 2)
    y = to_c_contigous(y)
    t = to_c_contigous(t)
    lib.cost_terms_trace(R, y[...], t[...], t.size)
    return R


def to_c_contigous(y: np.ndarray) -> np.ndarray:

    y = numpyfy(y)
    if not y.flags.c_contiguous:
        y = np.ascontiguousarray(y)
    return y


def cost_terms(
    v: np.ndarray,
    ca: np.ndarray,
    t_v: np.ndarray,
    t_ca: np.ndarray,
) -> np.ndarray:
    R = np.zeros(NUM_COST_TERMS)

    v = to_c_contigous(v)
    ca = to_c_contigous(ca)
    t_v = to_c_contigous(t_v)
    t_ca = to_c_contigous(t_ca)

    lib.cost_terms_trace(R[: NUM_COST_TERMS // 2], v[...], t_v[...], t_v.size)
    lib.cost_terms_trace(R[NUM_COST_TERMS // 2 :], ca[...], t_ca[...], t_ca.size)
    return R


def all_cost_terms(
    arr: np.ndarray,
    t: np.ndarray,
    mask: Optional[np.ndarray] = None,
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

    R = np.zeros((num_sets, NUM_COST_TERMS))
    lib.all_cost_terms(
        R,
        traces,
        t[...],
        mask[...],
        num_trace_points,
        num_sets,
        update_progress_func,
    )
    if progress_bar:
        progress_bar.close()
    return R
