import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

import numpy as np
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

Array = Union[np.ndarray, List[float], Sequence[float]]


class Backend(str, Enum):
    c = "c"
    numba = "numba"
    python = "python"


def _check_factor(factor: float) -> None:
    if not 0 < factor < 100:
        raise ValueError(f"Factor has to be between 0 and 100, got {factor}")
    if factor < 1:
        logger.warning(
            f"Factor passed to APD calculation is {factor}, did you mean {factor * 100}?",
        )


def intersection(
    data: Union[Sequence[Sequence[int]], Dict[Any, Sequence[int]]],
) -> Sequence[int]:
    """Get intersection of all values in
    a dictionary or a list of lists

    Parameters
    ----------
    data : Union[Sequence[Sequence[int]], Dict[str, Sequence[int]]]
        Input data

    Returns
    -------
    Sequence[int]
        The intersection
    """
    vals = data
    if isinstance(data, dict):
        vals = list(data.values())

    if len(vals) == 0:
        return []

    return list(set(vals[0]).intersection(*list(vals)))


def numpyfy(y) -> np.ndarray:
    if isinstance(y, (list, tuple)):
        y = np.array(y)

    try:
        import dask.array as da
    except ImportError:
        pass
    else:
        if isinstance(y, da.Array):
            y = y.compute()

    try:
        import h5py
    except ImportError:
        pass
    else:
        if isinstance(y, h5py.Dataset):
            y = y[...]

    return y


def normalize_signal(V, v_r=None):
    """
    Normalize signal to have maximum value 1
    and zero being the value equal to v_r (resting value).
    If v_r is not provided the minimum value
    in V will be used as v_r

    Arguments
    ---------
    V : array
        The signal
    v_r : float
        The resting value

    """

    # Maximum valu
    v_max = np.max(V)

    # Baseline or resting value
    if v_r is None:
        v_r = np.min(V)

    return (np.array(V) - v_r) / (v_max - v_r)


def time_unit(time_stamps):

    dt = np.mean(np.diff(time_stamps))
    # Assume dt is larger than 0.5 ms and smallar than 0.5 seconds
    unit = "ms" if dt > 0.5 else "s"
    return unit


def interpolate(t: np.ndarray, trace: np.ndarray, dt: float = 1.0):

    f = UnivariateSpline(t, trace, s=0, k=1)
    t0 = np.arange(t[0], t[-1] + dt, dt)
    return t0, f(t0)


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


def filt(y, kernel_size=None):
    """
    Filer signal using a median filter.
    Default kernel_size is 3
    """
    if kernel_size is None:
        kernel_size = 3

    logger.debug("\nFilter image")
    from scipy.signal import medfilt

    smooth_trace = medfilt(y, kernel_size)
    return smooth_trace
