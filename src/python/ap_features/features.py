import logging
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import _c, _numba
from .utils import Array, Backend, _check_factor

logger = logging.getLogger(__name__)

Upstroke = namedtuple(
    "Upstroke",
    ["index", "value", "upstroke", "dt", "x0", "sigmoid", "time_APD20_to_APD80"],
)
APDCoords = namedtuple("APDCoords", ["x1", "x2", "y1", "y2", "yth"])


def apd(
    factor: int,
    V: Array,
    t: Array,
    v_r: Optional[float] = None,
    use_spline: bool = True,
    backend: Backend = Backend.python,
) -> Optional[float]:
    r"""Return the action potential duration at the given
    factor repolarization, so that factor = 0
    would be zero, and factor = 1 given the time from triggering
    to potential is down to resting potential.

    Parameters
    ----------
    factor : int
        The APD factor between 0 and 100
    V : Array
        The signal
    t : Array
        The time stamps
    v_r : Optional[float], optional
        Resting value, by default None. Only applicablle for python Backend.
    use_spline : bool, optional
        Use spline interpolation, by default True.
        Only applicablle for python Backend.
    backend : Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'


    Returns
    -------
    float
        The action potential duration

    .. Note::

        If the signal has more intersection than two with the
        APX_X line, then the first and second occurence will be used.


    Raises
    ------
    ValueError
        If factor is outside the range of (0, 100)
    ValueError
        If shape of x and y does not match
    RunTimeError
        In unable to compute APD

    Notes
    -----
    If the signal represent voltage, we would like to compute the action
    potential duration at a given factor of the signal. Formally,
    Let :math:`p \in (0, 100)`, and define

    .. math::
        y_p = \tilde{y} - \left(1- \frac{p}{100} \right).

    Now let :math:`\mathcal{T}` denote the set of solutions of :math:`y_p = 0`,
    i.e

    .. math::
        y_p(t) = 0, \;\; \forall t \in \mathcal{T}

    Then there are three different scenarios; :math:`\mathcal{T}` contains one,
    two or more than two elements. Note that :math:`\mathcal{T}` cannot be
    empty (because of the intermediate value theorem).
    The only valid scenario is the case when :math:`\mathcal{T}` contains
    two (or more) elements. In this case we define

    .. math::
        \mathrm{APD} \; p = \max \mathcal{T} - \min \mathcal{T}

    for :math:`p < 0.5` and

    .. math::
        \mathrm{APD} \; p = \min \mathcal{T} / \min \mathcal{T}  - \min \mathcal{T}

    """ ""

    assert backend in Backend.__members__

    _check_factor(factor)

    y = np.array(V)
    x = np.array(t)
    if y.shape != x.shape:
        raise ValueError("The signal and time are not of same lenght")

    if backend == Backend.python:
        try:
            x1, x2 = _apd(factor=factor, V=y, t=x, v_r=v_r, use_spline=use_spline)
        except RuntimeError:
            return None
        return x2 - x1
    elif backend == Backend.c:
        return _c.apd(y=y, t=x, factor=factor)
    else:
        return _numba.apd(V=y, T=x, factor=factor)


def _apd(
    factor: int,
    V: np.ndarray,
    t: np.ndarray,
    v_r: Optional[float] = None,
    use_spline=True,
) -> Tuple[float, float]:

    _check_factor(factor)
    y = normalize_signal(V, v_r) - (1 - factor / 100)

    if use_spline:

        try:
            f = UnivariateSpline(t, y, s=0, k=3)
        except Exception as ex:
            msg = f"Unable to compute APD {factor * 100}. Please change your settings, {ex}"
            logger.warning(msg)
            raise RuntimeError(msg)

        inds = f.roots()
        if len(inds) == 1:
            # Safety guard for strange interpolations
            inds = t[np.where(np.diff(np.sign(y)))[0]]
    else:
        inds = t[np.where(np.diff(np.sign(y)))[0]]
    print(inds)
    if len(inds) == 0:
        logger.warning("Warning: no root was found for APD {}".format(factor))
        x1 = x2 = 0
    if len(inds) == 1:
        x1 = x2 = inds[0]
        logger.warning("Warning: only one root was found for APD {}" "".format(factor))
    else:
        start_index = int(np.argmax(np.diff(inds)))
        x1 = sorted(inds)[start_index]
        x2 = sorted(inds)[start_index + 1]

    return x1, x2


def apd_coords(
    factor: int, V: Array, t: Array, v_r: Optional[float] = None, use_spline=True,
) -> APDCoords:
    """Return the coordinates of the start and stop
        of the APD, and not the duration itself

    Parameters
    ----------
    factor : int
        The APD factor between 0 and 100
    V : Array
        The signal
    t : Array
        The time stamps
    v_r : Optional[float], optional
        Resting value, by default None. Only applicablle for python Backend.
    use_spline : bool, optional
        Use spline interpolation, by default True.
        Only applicablle for python Backend.

    Returns
    -------
    APDCoords
        APD coordinates
    """
    _check_factor(factor)
    y = np.array(V)
    x = np.array(t)
    x1, x2 = _apd(factor=factor, V=y, t=x, v_r=v_r, use_spline=use_spline)
    g = UnivariateSpline(x, y, s=0)
    y1 = g(x1)
    y2 = g(x2)

    yth = np.min(y) + (1 - factor) * (np.max(y) - np.min(y))

    return APDCoords(x1, x2, y1, y2, yth)


def tau(
    x: Array, y: Array, a: float = 0.75, backend: Backend = Backend.python,
) -> float:
    """
    Decay time. Time for the signal amplitude to go from maxium to
    (1 - a) * 100 % of maximum

    Parameters
    ----------
    x : Array
        The time stamps
    y : Array
        The signal
    a : float, optional
        The value for which you want to estimate the time decay, by default 0.75
    backend : Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float
        Decay time

    Raises
    ------
    NotImplementedError
        If unsupported backend is used
    """
    if backend != Backend.python:
        raise NotImplementedError(
            "Method currently only implemented for python backend"
        )

    Y = UnivariateSpline(x, normalize_signal(y) - a, s=0, k=3)
    t_max = x[int(np.argmax(y))]
    r = Y.roots()
    if len(r) >= 2:
        t_a = r[1]
    elif len(r) == 1:
        logger.warning(
            (
                "Only one zero was found when computing tau{}. " "Result might be wrong"
            ).format(int(a * 100))
        )
        t_a = r[0]
    else:
        logger.warning(
            (
                "No zero found when computing tau{}. "
                "Return the value of time to peak"
            ).format(int(a * 100))
        )
        t_a = x[0]

    return t_a - t_max


def time_to_peak(
    x: Array,
    y: Array,
    pacing: Optional[Array] = None,
    backend: Backend = Backend.python,
) -> float:
    """Computed the time to peak from pacing is
    triggered to maximum amplitude. Note, if pacing
    is not provided it will compute the time from
    the beginning of the trace (which might not be consistent)
    to the peak.

    Parameters
    ----------
    x : Array
        The time stamps
    y : Array
        The signal
    pacing : Optional[Array], optional
        The pacing amplitude, by default None
    backend : Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float
        Time to peak

    Raises
    ------
    NotImplementedError
        If unsupported backend is used
    """
    if backend != Backend.python:
        raise NotImplementedError(
            "Method currently only implemented for python backend"
        )

    if pacing is None:
        return x[int(np.argmax(y))]

    t_max = x[int(np.argmax(y))]
    if pacing is None:
        t_start = x[0]
    else:
        try:
            start_idx = (
                next(
                    i
                    for i, p in enumerate(np.diff(np.array(pacing).astype(float)))
                    if p > 0
                )
                + 1
            )
        except StopIteration:
            start_idx = 0
    t_start = x[start_idx]

    return t_max - t_start


def upstroke(
    x: Array, y: Array, a: float = 0.8, backend: Backend = Backend.python,
) -> float:
    """Compute the time from (1-a)*100 % signal
    amplitude to peak. For example if if a = 0.8
    if will compute the time from the starting value
    of APD80 to the upstroke.

    Parameters
    ----------
    x : Array
        The time stamps
    y : Array
        The signal
    a : float, optional
        Fraction of signal amplitide, by default 0.8
    backend : Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float
        The upstroke value

    Raises
    ------
    NotImplementedError
        If unsupported backend is used
    ValueError
        If a is outside the range of (0, 1)
    """
    if backend != Backend.python:
        raise NotImplementedError(
            "Method currently only implemented for python backend"
        )

    if not 0 < a < 1:
        raise ValueError("'a' has to be between 0.0 and 1.0")

    Y = UnivariateSpline(x, normalize_signal(y) - (1 - a), s=0, k=3)
    t_max = x[int(np.argmax(y))]
    r = Y.roots()
    if len(r) >= 1:
        if len(r) == 1:
            logger.warning(
                (
                    "Only one zero was found when computing upstroke{}. "
                    "Result might be wrong"
                ).format(int(a * 100))
            )
        t_a = r[0]
    else:
        logger.warning(
            (
                "No zero found when computing upstroke{}. "
                "Return the value of time to peak"
            ).format(int(a * 100))
        )
        t_a = x[0]

    return t_max - t_a


def beating_frequency(times: List[Array], unit: str = "ms") -> float:
    """Returns the approximate beating frequency in Hz by
    finding the average duration of each beat

    Parameters
    ----------
    times : List[Array]
        Time stamps of all beats
    unit : str, optional
        Unit of time, by default "ms"

    Returns
    -------
    float
        Beating frequency in Hz
    """
    if len(times) == 0:
        return np.nan
    # Get chopped data
    # Find the average lenght of each beat in time
    t_mean = np.mean([ti[-1] - ti[0] for ti in times])
    # Convert to seconds
    if unit == "ms":
        t_mean /= 1000.0
    # Return the freqency
    return 1.0 / t_mean


def beating_frequency_from_peaks(
    signals: List[Array], times: List[Array], unit: str = "ms",
) -> float:
    """Returns the beating frequency in Hz by using
    the peak values of the signals in each beat

    Parameters
    ----------
    signals : List[Array]
        The signal values for each beat
    times : List[Array]
        The time stamps of all beats
    unit : str, optional
        Unit of time, by default "ms"

    Returns
    -------
    float
        Beating frequency in Hz
    """

    t_maxs = [t[int(np.argmax(c))] for c, t in zip(signals, times)]

    dt = np.diff(t_maxs)
    if unit == "ms":
        dt = np.divide(dt, 1000.0)
    return np.divide(1, dt)


def find_upstroke_values(
    t: np.ndarray, y: np.ndarray, upstroke_duration: int = 50, normalize: bool = True
) -> np.ndarray:

    # Find intersection with APD50 line
    y_mid = (np.max(y) + np.min(y)) / 2
    f = UnivariateSpline(t, y - y_mid, s=0, k=3)
    zeros = f.roots()
    if len(zeros) == 0:
        return np.array([])
    idx_mid = next(i for i, ti in enumerate(t) if ti > zeros[0])

    # Upstroke should not be more than 50 ms
    N = upstroke_duration // 2
    upstroke = y[idx_mid - N : idx_mid + N + 1]

    if normalize and len(upstroke) > 0:
        from scipy.ndimage import gaussian_filter1d

        upstroke_smooth = gaussian_filter1d(upstroke, 2.0)

        y_min = np.min(upstroke_smooth)
        y_max = np.max(upstroke_smooth)

        upstroke_normalized = (upstroke - y_min) / (y_max - y_min)

        return upstroke_normalized

    return upstroke


def time_between_APDs(
    t: Array, y: Array, from_APD: int, to_APD: int, backend: Backend = Backend.python
):
    """Find the duration between first intersection (i.e
    during the upstroke) of two APD lines

    Arguments
    ---------
    t : np.ndarray
        Time values
    y : np.ndarray
        The trace
    from_APD: int
        First APD line
    to_APD: int
        Second APD line

    Returns
    -------
    float:
        The time between `from_APD` to `to_APD`

    Example
    -------

    .. code:: python

        # Compute the time between APD20 and APD80
        t2080 = time_between_APDs(t, y, 20, 80)

    """
    _check_factor(from_APD)
    _check_factor(to_APD)

    y_norm = normalize_signal(y)

    y_from = UnivariateSpline(t, y_norm - from_APD / 100, s=0, k=3)
    t_from = y_from.roots()[0]

    y_to = UnivariateSpline(t, y_norm - to_APD / 100, s=0, k=3)
    t_to = y_to.roots()[0]

    return t_to - t_from


def max_relative_upstroke_velocity(
    t: np.ndarray, y: np.ndarray, upstroke_duration: int = 50, sigmoid_fit: bool = True
) -> Upstroke:
    """Estimate maximum relative upstroke velocity

    Arguments
    ---------
    t : np.ndarray
        Time values
    y : np.ndarray
        The trace
    upstroke_duration : int
        Duration in milliseconds of upstroke (Default: 50).
        This does not have to be exact up should at least be
        longer than the upstroke.
    sigmoid_fit : bool
        If True then use a sigmoid function to fit the data
        of the upstroke and report the maximum derivate of
        the sigmoid as the maximum upstroke.

    Notes
    -----
    Brief outline of current algorithm:
    1. Interpolate data to have time resolution of 1 ms.
    2. Find first intersection with ADP50 line
    3. Select 25 ms before and after the point found in 2
    4. Normalize the 50 ms (upstroke_duration) from 3 so that we have
    a max and min value of 1 and 0 respectively.
    If sigmoid fit is True
    Fit the normalize trace to a sigmoid function and compte the

    else:
    5. Compute the successive differences in amplitude (i.e delta y)
    and report the maximum value of these


    """

    # Interpolate to 1ms precision
    t0, y0 = interpolate(t, y, dt=1.0)
    # Find values beloning to upstroke
    upstroke = find_upstroke_values(t0, y0, upstroke_duration=upstroke_duration)
    dt = np.mean(np.diff(t))
    if len(upstroke) == 0:
        # There is no signal
        index = 0
        value = np.nan
        x0 = None
        s = None
        time_APD20_to_APD80 = np.nan
    else:
        t_upstroke = t0[: len(upstroke)]
        t_upstroke -= t_upstroke[0]
        if sigmoid_fit:

            def sigmoid(x, k, x0):
                return 0.5 * (np.tanh(k * (x - x0)) + 1)

            from scipy.optimize import curve_fit

            popt, pcov = curve_fit(sigmoid, t_upstroke, upstroke, method="dogbox")

            k, x0 = popt
            index = None  # type:ignore
            s = sigmoid(t_upstroke, k, x0)
            value = k / 2
            time_APD20_to_APD80 = time_between_APDs(t_upstroke, s, 20, 80)

        else:
            # Find max upstroke
            index = np.argmax(np.diff(upstroke))  # type:ignore
            value = np.max(np.diff(upstroke))
            x0 = None
            s = None
            time_APD20_to_APD80 = time_between_APDs(t_upstroke, upstroke, 20, 80)

    return Upstroke(
        index=index,
        value=value,
        upstroke=upstroke,
        dt=dt,
        x0=x0,
        sigmoid=s,
        time_APD20_to_APD80=time_APD20_to_APD80,
    )


def maximum_upstroke_velocity(y, t=None, use_spline=True, normalize=False):
    r"""
    Compute maximum upstroke velocity

    Arguments
    ---------
    y : array
        The signal
    t : array
        The time points
    use_spline : bool
        Use spline interpolation
        (Default : True)
    normalize : bool
        If true normalize signal first, so that max value is 1.0,
        and min value is zero before performing the computation.

    Returns
    -------
    float
        The maximum upstroke velocity

    Notes
    -----
    If :math:`y` is the voltage, this feature corresponds to the
    maximum upstroke velocity. In the case when :math:`y` is a continuous
    signal, i.e a 5th order spline interpolant of the data, then we can
    simply find the roots of the second derivative, and evaluate the
    derivative at that point, i.e

    .. math::
        \max \frac{\mathrm{d}y}{\mathrm{d}t}
        = \frac{\mathrm{d}y}{\mathrm{d}t} (t^*),

    where

    .. math::
        \frac{\mathrm{d}^2y}{\mathrm{d}^2t}(t^*) = 0.

    If we decide to use the discrete version of the signal then the
    above each derivative is approximated using a forward difference, i.e

    .. math::
        \frac{\mathrm{d}y}{\mathrm{d}t}(t_i) \approx
        \frac{y_{i+1}-y_i}{t_{i+1} - t_i}.

    """

    if t is None:
        t = range(len(y))

    msg = "The signal and time are not of same lenght"
    assert len(t) == len(y), msg

    # Normalize
    if normalize:
        y = normalize_signal(y)

    if use_spline:
        f = UnivariateSpline(t, y, s=0, k=5)
        h = f.derivative()
        max_upstroke_vel = np.max(h(h.derivative().roots()))
    else:
        max_upstroke_vel = np.max(np.divide(np.diff(y), np.diff(t)))

    return max_upstroke_vel


def integrate_apd(y, t=None, percent=0.3, use_spline=True, normalize=False):
    r"""
    Compute the integral of the signals above
    the APD p line

    Arguments
    ---------
    y : array
        The signal
    t : array
        The time points
    use_spline : bool
        Use spline interpolation
        (Default : True)
    normalize : bool
        If true normalize signal first, so that max value is 1.0,
        and min value is zero before performing the computation.


    Returns
    -------
    integral : float
        The integral above the line defined by the APD p line.

    Notes
    -----
    This feature represents the integral of the signal above the
    horizontal line defined by :math:`\mathrm{APD} p`. First let

    .. math::
        y_p = \tilde{y} - \left(1- \frac{p}{100} \right),

    and let :math:`t_1` and :math:`t_2` be the two solutions solving
    :math:`y_p(t_i) = 0, i=1,2` (assume that we have 2 solutions only).
    The integral we are seeking can now be computed as follows:

    .. math::
        \mathrm{Int} \; p = \int_{t_1}^{t_2}
        \left[ y - y(t_1) \right] \mathrm{d}t

    """

    if normalize:
        y = normalize_signal(y)

    if t is None:
        t = range(len(y))

    coords, vals = apd(y, percent, t, return_coords=True, use_spline=use_spline)

    if use_spline:
        Y = y - vals[0]
        f = UnivariateSpline(t, Y, s=0, k=3)
        integral = f.integral(*coords)

    else:
        Y = y - vals[-1]

        t1 = t.tolist().index(coords[0])
        t2 = t.tolist().index(coords[1]) + 1

        integral = np.sum(np.multiply(Y[t1:t2], np.diff(t)[t1:t2]))

    return integral


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


def corrected_apd(apd, beat_rate, formula="friderica"):
    """Correct the given APD (or any QT measurement) for the beat rate.
    normally the faster the HR (or the shorter the RR interval),
    the shorter the QT interval, and vice versa

    Friderica formula (default):

    .. math::

        APD (RR)^{-1/3}

    Bazett formula:

    .. math::

        APD (RR)^{-1/2}

    """

    formulas = ["friderica", "bazett"]
    msg = f"Expected formula to be one of {formulas}, got {formula}"
    assert formula in formulas, msg
    RR = np.divide(60, beat_rate)

    if formula == "friderica":
        return np.multiply(apd, pow(RR, -1 / 3))
    else:
        return np.multiply(apd, pow(RR, -1 / 2))
