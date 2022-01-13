import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from . import utils
from .utils import Array

logger = logging.getLogger(__name__)


class InvalidFilter(RuntimeError):
    pass


class Filters(str, Enum):
    apd30 = "apd30"
    apd50 = "apd50"
    apd70 = "apd70"
    apd80 = "apd80"
    apd90 = "apd90"
    length = "length"
    time_to_peak = "time_to_peak"


def filter_signals(
    data: Union[Sequence[Array], Dict[Any, Array]],
    x: float = 1,
    center="mean",
) -> Sequence[int]:

    if len(data) == 0:
        return []

    values = data
    if isinstance(data, dict):
        values = list(data.values())

    # Check that all arrays have the same length
    v0 = values[0]
    N = len(v0)
    for v in values:
        if len(v) != N:
            raise RuntimeError("Unequal length of arrays")

    all_indices = []
    for v in values:
        indices = within_x_std(v, x, center)
        # If no indices are with the tolerace then we just
        # include everything
        all_indices.append(indices if len(indices) > 0 else np.arange(N))

    return utils.intersection(all_indices)


def within_x_std(arr: Array, x: float = 1.0, center="mean") -> Sequence[int]:
    """Get the indices in the array that are
    within x standard deviations from the center value

    Parameters
    ----------
    arr : Array
        The array with values
    x : float, optional
        Number of standard deviations, by default 1.0
    center : str, optional
        Center value, Either "mean" or "median", by default "mean"

    Returns
    -------
    Sequence[int]
        Indices of the values that are within x
        standard deviations from the center value
    """
    if len(arr) == 0:
        return []

    msg = f"Expected 'center' to be 'mean' or 'median', got {center}"
    assert center in ["mean", "median"], msg
    mu = np.mean(arr) if center == "mean" else np.median(arr)

    std = np.std(arr)
    within = [abs(a - mu) <= x * std for a in arr]
    return np.where(within)[0]


def filt(y: Array, kernel_size: int = 3):
    """
    Filer signal using a median filter.
    Default kernel_size is 3
    """

    logger.debug("Filter image")
    from scipy.signal import medfilt

    smooth_trace = medfilt(y, kernel_size)
    return smooth_trace


def remove_points(
    x: Array,
    y: Array,
    t_start: float,
    t_end: float,
    normalize: bool = True,
) -> Tuple[Array, Array]:
    """
    Remove points in x and y between start and end.
    Also make sure that the new x starts a zero if
    normalize = True
    """
    if not len(x) == len(y):
        raise ValueError(
            f"Expected x and y to have same length, got len(x) = {len(x)} and len(y) = {len(y)}",
        )

    start = next(i for i, t in enumerate(x) if t > t_start) - 1
    try:
        end = next(i for i, t in enumerate(x) if t > t_end)
    except StopIteration:
        end = len(x) - 1

    logger.debug(
        ("Remove points for t={} (index:{}) to t={} (index:{})" "").format(
            t_start,
            start,
            t_end,
            end,
        ),
    )
    x0 = x[:start]
    x1 = np.subtract(x[end:], x[end] - x[start])
    x_new = np.concatenate((x0, x1))

    if normalize:
        x_new -= x_new[0]
    y_new = np.concatenate((y[:start], y[end:]))

    return x_new, y_new


def find_spike_points(pacing, spike_duration: int = 7) -> List[int]:
    """
    Remove spikes from signal due to pacing.

    We assume that there is a spike starting at the
    time of pacing and the spike dissapears after
    some duration. These points will be deleted
    from the signal

    Parameters
    ----------
    pacing : array
        The pacing amplitude of same length as y
    spike_duration: int
        Duration of the spikes

    Returns
    -------
    np.ndarray
        A list of indices containing the spikes

    """
    if spike_duration == 0:
        return []

    # Find time of pacing
    (inds,) = np.where(np.diff(np.array(pacing, dtype=float)) > 0)

    if len(inds) == 0:
        logger.warning("No pacing found. Spike removal not possible.")
        return []

    spike_points = np.concatenate(
        [np.arange(i, i + spike_duration) for i in inds],
    ).astype(int)

    return spike_points
