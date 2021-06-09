import logging

import numpy as np

logger = logging.getLogger(__name__)


def correct_background(x, y, *args, **kwargs):
    r"""
    Perform at background correction.
    First estimate background :math:`b`, and let
    :math:`F_0 = b(0)`. The corrected background is
    then :math:`\frac{y - b}{F_0}`

    Arguments
    ---------
    x : np.mdarray
        Time points
    y : Fluorecense amplitude

    Returns
    -------
    np.mdarrray
        The corrected trace
    """

    bkg = background(x, y, *args, **kwargs)
    F0 = bkg[0]
    corrected = (1 / F0) * (y - bkg)
    return corrected


def background(x, y, order=2, s=0.01, fct="atq"):
    """
    Compute an estimation of the background (aka baseline)
    in chemical spectra


    This is a reimplementation of a MATLAB script that
    can be found `here <https://se.mathworks.com/matlabcentral
    /fileexchange/27429-background-correction>`_

    .. rubric:: References

    [1] Mazet, V., Carteret, C., Brie, D., Idier, J. and Humbert, B.,
    2005. Background removal from spectra by designing and minimising
    a non-quadratic cost function. Chemometrics and intelligent
    laboratory systems, 76(2), pp.121-133.

    """

    # Rescaling
    N = len(x)
    x, i = np.sort(x), np.argsort(x)

    y = y[i]

    maxy = np.max(y)
    dely = (maxy - np.min(y)) / 2
    # Normalize time
    x = 2 * np.divide(np.subtract(x, x[N - 1]), x[N - 1] - x[0]) + 1
    y = np.subtract(y, maxy) / dely + 1

    # Vandermonde matrix
    T = np.vander(x, order + 1)
    Tinv = np.linalg.pinv(T.T.dot(T)).dot(T.T)

    # Initialisation (least-squares estimation)
    a = Tinv.dot(y)
    z = T.dot(a)

    #  Other variables
    alpha = 0.99 * 1 / 2  # Scale parameter alpha
    it = 0  # Iteration number
    zp = np.ones(N)  # Previous estimation

    # Iterate
    while np.sum((z - zp) ** 2) / np.sum(zp ** 2) > 1e-9:

        it = it + 1  # Iteration number
        zp = z  # Previous estimation
        res = y - z  # Residual

        d = np.zeros(len(res))

        # Estimate d
        if fct == "sh":
            d[np.abs(res) < s] += res[np.abs(res) < s] * (2 * alpha - 1)
            d[res <= -s] -= alpha * 2 * s + res[res <= -s]
            d[res >= s] -= res[res >= s] - alpha * 2 * s

        elif fct == "ah":
            d[np.abs(res) < s] += res[np.abs(res) < s] * (2 * alpha - 1)
            d[res >= s] -= res[res >= s] - alpha * 2 * s
        elif fct == "stq":
            d[np.abs(res) < s] += res[np.abs(res) < s] * (2 * alpha - 1)
            d[res >= s] -= res[res >= s] - alpha * 2 * s

        elif fct == "atq":
            d[res < s] += res[res < s] * (2 * alpha - 1)
            d[res >= s] -= res[res >= s]

        # Estimate z
        a = Tinv.dot(y + d)  # Polynomial coefficients a
        z = T.dot(a)  # Polynomial

    # Rescaling
    j = np.argsort(i)
    z = (z[j] - 1) * dely + maxy

    a[0] = a[0] - 1
    a = a * dely  # + maxy
    return z
