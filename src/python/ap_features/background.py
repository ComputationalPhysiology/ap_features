import logging
from enum import Enum
from typing import NamedTuple

import numpy as np
from scipy.signal import medfilt

from .utils import Array


logger = logging.getLogger(__name__)


class BackgroundCorrection(str, Enum):
    full = "full"
    subtract = "subtract"
    none = "none"


class BackgroundCostFunction(str, Enum):
    sh = "sh"
    ah = "ah"
    stq = "stq"
    atq = "atq"


class Background(NamedTuple):
    x: Array
    y: Array
    y_filt: Array
    corrected: Array
    background: Array
    F0: float
    method: BackgroundCorrection


def get_filtered_signal(y: Array, filter_kernel_size: int = 0) -> Array:
    if filter_kernel_size == 0:
        return y
    else:
        return medfilt(y, kernel_size=13)


def correct_background(
    x: Array,
    y: Array,
    method: BackgroundCorrection,
    filter_kernel_size: int = 0,
    **kwargs,
) -> Background:
    methods = tuple(BackgroundCorrection.__members__.keys())
    if method not in methods:
        raise ValueError(f"Invalid method '{method}', expected one of {methods}")

    if len(x) != len(y):
        raise ValueError(f"Size of x ({len(x)}) and y ({len(y)}) did not match")

    y_filt = get_filtered_signal(y, filter_kernel_size=filter_kernel_size)

    if method == BackgroundCorrection.none:
        return Background(
            x=x,
            y=y,
            y_filt=y_filt,
            corrected=y,
            background=np.zeros_like(y),
            F0=1,
            method=method,
        )

    bkg = background(x, y_filt, **kwargs)

    if method == BackgroundCorrection.full:
        F0 = bkg[0]

    if method == BackgroundCorrection.subtract:
        F0 = 1
    corrected = (1 / F0) * (y - bkg)
    return Background(
        x=x,
        y=y,
        y_filt=y_filt,
        corrected=corrected,
        background=bkg,
        F0=F0,
        method=method,
    )


def full_background_correction(x: Array, y: Array, filter_kernel_size=0, **kwargs) -> Background:
    r"""Perform at background correction.
    First estimate background :math:`b`, and let
    :math:`F_0 = b(0)`. The corrected background is
    then :math:`\frac{y - b}{F_0}`. Additional argument
    can be passed to background corrected algorithm as
    keyword arguments.

    Parameters
    ----------
    x : Array
        Time points
    y : Array
        Fluorescence amplitude

    Returns
    -------
    Background
        Namedtuple containing the corrected trace and the background.
    """
    y_filt = get_filtered_signal(y, filter_kernel_size=filter_kernel_size)

    bkg = background(x, y_filt, **kwargs)
    F0 = bkg[0]
    corrected = (1 / F0) * (y - bkg)
    return Background(
        x=x,
        y=y,
        y_filt=y_filt,
        corrected=corrected,
        background=bkg,
        F0=F0,
        method=BackgroundCorrection.full,
    )


def background(
    x: Array,
    y: Array,
    order: int = 2,
    threshold: float = 0.01,
    cost_function: BackgroundCostFunction = BackgroundCostFunction.atq,
    **kwargs,
) -> np.ndarray:
    r"""Compute an estimation of the background (aka baseline)
    in chemical spectra. The background is estimated by a polynomial
    with order using a cost-function and a threshold parameter.
    This is a re-implementation of a MATLAB script that
    can be found `here <https://se.mathworks.com/matlabcentral
    /fileexchange/27429-background-correction>`_

    Parameters
    ----------
    x : Array
        Time stamps
    y : Array
        Signal
    order : int, optional
        Polynomial order, by default 2
    threshold : float, optional
        The threshold parameters, by default 0.01
    cost_function : BackgroundCostFunction, optional
        Cost function to be minimized, by default BackgroundCostFunction.atq.
        The cost functions can have the following forms:

    Returns
    -------
    np.ndarray
        The estimated baseline

    Notes
    -----

    The cost function can have the four following values:

    * sh  - symmetric Huber function :

        .. math::
            f(x) =  \begin{cases}
                    x^2, \; \text{ if } |x| < \text{threshold} \\
                    2 \text{threshold}  |x|-\text{threshold}^2,  \; \text{otherwise}
                    \end{cases}

    * ah  - asymmetric Huber function :

        .. math::
            f(x) =  \begin{cases}
                    x^2, \; \text{ if } x < \text{threshold} \\
                    2 \text{threshold} x-\text{threshold}^2 , \; \text{otherwise}
                    \end{cases}


    * stq - symmetric truncated quadratic :

        .. math::
            f(x) =  \begin{cases}
                    x^2, \; \text{ if } |x| < \text{threshold} \\
                    \text{threshold}^2 , \; \text{otherwise}
                    \end{cases}

    * atq - asymmetric truncated quadratic :

        .. math::
            f(x) =  \begin{cases}
                    x^2, \; \text{ if } x < \text{threshold} \\
                    \text{threshold}^2 , \; \text{otherwise}
                    \end{cases}


    .. rubric:: References

    [1] Mazet, V., Carteret, C., Brie, D., Idier, J. and Humbert, B.,
    2005. Background removal from spectra by designing and minimising
    a non-quadratic cost function. Chemometrics and intelligent
    laboratory systems, 76(2), pp.121-133.
    """

    # Rescaling
    N = len(x)
    x_sorted, i = np.sort(x), np.argsort(x).astype(int)
    y_sorted = np.array(y)[i]

    maxy = np.max(y_sorted)
    dely = (maxy - np.min(y_sorted)) / 2
    # Normalize time
    x_norm = (
        2
        * np.divide(
            np.subtract(x_sorted, x_sorted[N - 1]),
            x_sorted[N - 1] - x_sorted[0],
        )
        + 1
    )
    y_norm = np.subtract(y_sorted, maxy) / dely + 1

    # Vandermonde matrix
    T = np.vander(x_norm, order + 1)
    Tinv = np.linalg.pinv(T.T.dot(T)).dot(T.T)

    # Initialization (least-squares estimation)
    a = Tinv.dot(y_norm)
    z = T.dot(a)

    #  Other variables
    alpha = 0.99 * 1 / 2  # Scale parameter alpha
    it = 0  # Iteration number
    zp = np.ones(N)  # Previous estimation

    # Iterate
    while np.sum((z - zp) ** 2) / np.sum(zp**2) > 1e-9:
        it = it + 1  # Iteration number
        zp = z  # Previous estimation
        res = y_norm - z  # Residual

        d = np.zeros(len(res))

        # Estimate d
        if cost_function == BackgroundCostFunction.sh:
            d[np.abs(res) < threshold] += res[np.abs(res) < threshold] * (2 * alpha - 1)
            d[res <= -threshold] -= alpha * 2 * threshold + res[res <= -threshold]
            d[res >= threshold] -= res[res >= threshold] - alpha * 2 * threshold

        elif cost_function == BackgroundCostFunction.ah:
            d[np.abs(res) < threshold] += res[np.abs(res) < threshold] * (2 * alpha - 1)
            d[res >= threshold] -= res[res >= threshold] - alpha * 2 * threshold
        elif cost_function == BackgroundCostFunction.stq:
            d[np.abs(res) < threshold] += res[np.abs(res) < threshold] * (2 * alpha - 1)
            d[res >= threshold] -= res[res >= threshold] - alpha * 2 * threshold

        elif cost_function == BackgroundCostFunction.atq:
            d[res < threshold] += res[res < threshold] * (2 * alpha - 1)
            d[res >= threshold] -= res[res >= threshold]

        # Estimate z
        a = Tinv.dot(y_norm + d)  # Polynomial coefficients a
        z = T.dot(a)  # Polynomial

    # Rescaling
    j = np.argsort(i)
    z = (z[j] - 1) * dely + maxy

    a[0] = a[0] - 1
    a = a * dely  # + maxy
    return z
