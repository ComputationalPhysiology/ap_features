from collections import namedtuple
from itertools import zip_longest as izip_longest
from typing import List
from typing import Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

from .utils import Array

Average = namedtuple("Average", ["y", "x", "ys", "xs"])


def average_list(signals: List[Array]) -> Array:
    """Get average of signals.
    Assume that signals are alinged, but
    they dont have to be of the same length.
    If they have different length then the output
    average will have the same lenght as the longest
    array

    Parameters
    ----------
    signals : List[Array]
        The data that you want to average

    Returns
    -------
    Array
        The average signal
    """

    if len(signals) == 0:
        return []

    if len(signals) == 1:
        return signals[0]

    # Check is they have the same lenght
    if all([len(s) == len(signals[0]) for s in signals[1:]]):
        # Then it is easy to take the average
        average = np.mean(signals, 0)

    else:
        # We need to take into account the possibilty
        # the the subsignals have different lenght
        def avg(x):
            x = [i for i in x if i]
            if len(x) == 0:
                return 0.0
            return sum(x, 0.0) / len(x)

        average = np.array(tuple(map(avg, izip_longest(*signals))))

    return average


def average_and_interpolate(
    ys: List[Array],
    xs: Optional[List[Array]] = None,
    N: int = 200,
):
    """
    Get the avagere of list of signals assuming that
    they align at the same x value

    Parameters
    ----------
    ys : Array
        The signal values
    xs : Array
        The x-values
    N : int
        Lenght of output array (Default: 200)

    Returns
    -------
    Y_avg: array
        The average y values
    X : array
        The new x-values

    """
    if xs is None or len(xs) == 0:
        return average_list(ys), np.arange(max([len(i) for i in ys]))

    # Construct new time array
    min_x = np.min([xi[0] for xi in xs])
    max_x = np.max([xi[-1] for xi in xs])
    X = np.linspace(min_x, max_x, N)

    if len(ys) == 0:
        return np.zeros(N), X

    # Check args
    msg = (
        "Expected Xs and Ys has to be of same lenght. " "Got len(xs) = {}, len(ys) = {}"
    ).format(len(xs), len(ys))
    assert len(xs) == len(ys), msg

    for i, (x, y) in enumerate(zip(xs, ys)):
        msg = (
            "Expected X and Y has to be of same lenght. "
            "Got len(x) = {}, len(y) = {} for index {}"
        ).format(len(x), len(y), i)
        assert len(x) == len(y), msg

    Ys = []
    Xs = []

    for i, (x, y) in enumerate(zip(xs, ys)):
        # Take out relevant piece
        idx = next(j + 1 for j, xj in enumerate(X) if xj >= x[-1])

        X_ = X[:idx]
        Xs.append(X_)

        Y = UnivariateSpline(x, y, s=0)(X_)
        Ys.append(Y)

    Y_avg = average_list(Ys)

    return Average(y=Y_avg, x=X, ys=Ys, xs=Xs)
