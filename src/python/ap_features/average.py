from collections import namedtuple
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from .utils import Array

Average = namedtuple("Average", ["y", "x", "ys", "xs"])


class InvalidSubSignalError(RuntimeError):
    pass


def clean_data(
    ys: Sequence[Array],
    xs: Optional[Sequence[Array]],
) -> Tuple[Sequence[Array], Sequence[Array]]:
    """Make sure `xs` and `ys` have the
    correct shapes and remove empty sub-signals

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
    same as the order it is returned. Apart from
    this fact, the order doesn't matter.

    Raises
    ------
    InvalidSubSignalError
        If the length of `xs` and `ys` don't agree
    InvalidSubSignalError
        If the length of one of the sub-signals of `xs`
        and `ys` don't agree.
    """
    new_xs = []
    new_ys = []

    if xs is None:
        return ys, []

    if len(xs) != len(ys):
        raise InvalidSubSignalError(
            "Expected Xs and Ys has to be of same length. "
            f"Got len(xs) = {len(xs)}, len(ys) = {len(ys)}",
        )

    for i, (x, y) in enumerate(zip(xs, ys)):
        if len(x) != len(y):
            raise InvalidSubSignalError(
                "Expected X and Y has to be of same length. "
                f"Got len(x) = {len(x)}, len(y) = {len(y)} for index {i}",
            )
        if len(x) == 0:
            # Skip this one
            continue

        new_xs.append(x)
        new_ys.append(y)
    return new_ys, new_xs


def interpolate(X: Array, x: Array, y: Array) -> np.ndarray:
    """Interpolate array

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
    """Given a list of sub-signals create a new array of length
    `N` that cover all values

    Parameters
    ----------
    xs : Sequence[Array]
        List of monotonic sub sub-signal
    N : int
        Size of output array

    Returns
    -------
    np.ndarray
        Array that cover all values of length `N`
    """
    if len(xs) == 0:
        return np.arange(N)
    min_x = np.min([xi[0] for xi in xs])
    max_x = np.max([xi[-1] for xi in xs])
    return np.linspace(min_x, max_x, N)


def average_and_interpolate(
    ys: Sequence[Array],
    xs: Optional[Sequence[Array]] = None,
    N: int = 200,
) -> Average:
    """
    Get the average of list of signals assuming that
    they align at the same x value

    Parameters
    ----------
    ys : Array
        The signal values
    xs : Array
        The x-values
    N : int
        Length of output array (Default: 200)

    Returns
    -------
    Y_avg: array
        The average y values
    X : array
        The new x-values

    """

    ys, xs = clean_data(ys, xs)

    if len(xs) == 0:
        xs = [np.arange(0, len(yi)) for yi in ys]

    # Construct new time array
    X = create_longest_time_array(xs, N)

    if len(ys) == 0:
        return Average(y=np.zeros(N), x=X, xs=xs, ys=ys)

    if len(ys) == 1:
        return Average(y=interpolate(X, xs[0], ys[0]), x=X, xs=xs, ys=ys)

    Ys = []
    Xs = []

    Y = np.zeros_like(X)
    counts = np.zeros_like(X)

    for x, y in zip(xs, ys):
        # Take out relevant piece
        end_idx = next(j + 1 for j, xj in enumerate(X) if xj >= x[-1])
        start_idx = next(j for j, xj in enumerate(X) if xj >= x[0])

        xi = X[start_idx:end_idx]
        yi = interpolate(xi, x, y)

        Y[start_idx:end_idx] += yi
        counts[start_idx:end_idx] += 1

        Xs.append(xi)
        Ys.append(yi)

    Y_avg = np.divide(Y, counts, out=np.zeros_like(Y), where=counts != 0)

    return Average(y=Y_avg, x=X, ys=Ys, xs=Xs)
