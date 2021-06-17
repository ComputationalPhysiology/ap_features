import logging
from collections import namedtuple
from enum import Enum

import numpy as np

from .utils import Array

Background = namedtuple("Background", ["x", "y", "corrected", "background", "F0"])

logger = logging.getLogger(__name__)


class BackgroundCostFunction(str, Enum):
    sh = "sh"
    ah = "ah"
    stq = "stq"
    atq = "atq"


def correct_background(x: Array, y: Array, **kwargs) -> Background:
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
        Fluorecense amplitude

    Returns
    -------
    Background
        Namedtuple containing the corrected trace and the background.
    """

    bkg = background(x, y, **kwargs)
    F0 = bkg[0]
    corrected = (1 / F0) * (y - bkg)
    return Background(x=x, y=y, corrected=corrected, background=bkg, F0=F0)


def background(
    x: Array,
    y: Array,
    order: int = 2,
    threshold: float = 0.01,
    cost_function: BackgroundCostFunction = BackgroundCostFunction.atq,
) -> np.ndarray:
    r"""Compute an estimation of the background (aka baseline)
    in chemical spectra. The background is estimated by a polynomial
    with order using a cost-function and a threshold parameter.
    This is a reimplementation of a MATLAB script that
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

    # Initialisation (least-squares estimation)
    a = Tinv.dot(y_norm)
    z = T.dot(a)

    #  Other variables
    alpha = 0.99 * 1 / 2  # Scale parameter alpha
    it = 0  # Iteration number
    zp = np.ones(N)  # Previous estimation

    # Iterate
    while np.sum((z - zp) ** 2) / np.sum(zp ** 2) > 1e-9:

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
