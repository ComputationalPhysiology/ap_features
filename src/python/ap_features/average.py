from collections import namedtuple
from itertools import zip_longest as izip_longest
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from .utils import Array

Average = namedtuple("Average", ["y", "x", "ys", "xs"])


class InvalidSubSignalError(RuntimeError):
    pass


def average_list(signals: Sequence[Array]) -> Array:
    """Get average of signals.
    Assume that signals are alinged, but
    they dont have to be of the same length.
    If they have different length then the output
    average will have the same lenght as the longest
    array

    Parameters
    ----------
    signals : Sequence[Array]
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
        average = np.mean(signals, 0)  # type:ignore

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


def clean_data(
    ys: Sequence[Array],
    xs: Optional[Sequence[Array]],
) -> Tuple[Sequence[Array], Sequence[Array]]:
    """Make sure `xs` and `ys` have the
    correct shapes and remove empty subsignals

    Parameters
    ----------
    ys : Sequence[Array]
        First list
    xs : Optional[Sequence[Array]]
        Second list

    Returns
    -------
    Tuple[Sequence[Array], Sequence[Array]]
        (ys, xs) - cleaned version

    Note
    ----
    The order you send in the array will be the
    same as the order it is retured. Apart from
    this fact, the order doesn't matter.

    Raises
    ------
    InvalidSubSignalError
        If the length of `xs` and `ys` don't agree
    InvalidSubSignalError
        If the length of one of the subsignals of `xs`
        and `ys` don't agree.
    """
    new_xs = []
    new_ys = []

    if xs is None:
        return ys, []

    if len(xs) != len(ys):
        raise InvalidSubSignalError(
            "Expected Xs and Ys has to be of same lenght. "
            f"Got len(xs) = {len(xs)}, len(ys) = {len(ys)}",
        )

    for i, (x, y) in enumerate(zip(xs, ys)):
        if len(x) != len(y):
            raise InvalidSubSignalError(
                "Expected X and Y has to be of same lenght. "
                f"Got len(x) = {len(x)}, len(y) = {len(y)} for index {i}",
            )
        if len(x) == 0:
            # Skip this one
            continue

        new_xs.append(x)
        new_ys.append(y)
    return new_ys, new_xs


def interpolate(X: Array, x: Array, y: Array) -> np.ndarray:
    """Interapolate array

    Parameters
    ----------
    X : Array
        x-coordinates at which to evaluate the interpolated
        values
    x : Array
        x-coordinates of the data points
    y : Array
        y-coordinates of the data points

    Returns
    -------
    np.ndarray
        Interpolated y-coordinates

    Note
    ----
    This function will try to perform spline interpolation using
    `scipy.interpolate.UnivariateSpline` and fall back to
    `numpy.interp` in case that doesn't work
    """
    try:
        Y = UnivariateSpline(x, y, s=0)(X)
    except Exception:
        # https://stackoverflow.com/questions/64766510/catch-dfitpack-error-from-scipy-interpolate-interpolatedunivariatespline
        Y = np.interp(X, x, y)
    return Y


def create_longest_time_array(xs: Sequence[Array], N: int) -> np.ndarray:
    """Given a list of subsignals create a new array of length
    `N` that cover all values

    Parameters
    ----------
    xs : Sequence[Array]
        List of monotonic sub subsignal
    N : int
        Size of output arrayu

    Returns
    -------
    np.ndarray
        Array that cover all values of length `N`
    """
    min_x = np.min([xi[0] for xi in xs])
    max_x = np.max([xi[-1] for xi in xs])
    return np.linspace(min_x, max_x, N)


def average_and_interpolate(
    ys: Sequence[Array],
    xs: Optional[Sequence[Array]] = None,
    N: int = 200,
) -> Average:
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
    ys, xs = clean_data(ys, xs)

    if len(xs) == 0:
        y = average_list(ys)
        x = [] if len(ys) == 0 else np.arange(max([len(i) for i in ys]))
        return Average(y=y, x=x, xs=xs, ys=ys)

    # Construct new time array
    X = create_longest_time_array(xs, N)

    if len(ys) == 0:
        return Average(y=np.zeros(N), x=X, xs=xs, ys=ys)

    if len(ys) == 1:
        return Average(y=interpolate(X, xs[0], ys[0]), x=X, xs=xs, ys=ys)

    Ys = []
    Xs = []

    for i, (x, y) in enumerate(zip(xs, ys)):
        # Take out relevant piece
        idx = next(j + 1 for j, xj in enumerate(X) if xj >= x[-1])

        Xs.append(X[:idx])
        Ys.append(interpolate(Xs[-1], x, y))

    Y_avg = average_list(Ys)

    return Average(y=Y_avg, x=X, ys=Ys, xs=Xs)
