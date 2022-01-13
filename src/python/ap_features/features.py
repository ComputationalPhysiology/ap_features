import logging
from collections import namedtuple
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import _c
from . import _numba
from . import utils
from .utils import _check_factor
from .utils import Array
from .utils import Backend
from .utils import numpyfy

logger = logging.getLogger(__name__)

Upstroke = namedtuple(
    "Upstroke",
    ["index", "value", "upstroke", "dt", "x0", "sigmoid", "time_APD20_to_APD80"],
)
APDCoords = namedtuple("APDCoords", ["x1", "x2", "y1", "y2", "yth"])


class UnequalLengthError(RuntimeError):
    pass


def triangulation(
    V: Array,
    t: Array,
    low: int = 30,
    high: int = 80,
    v_r: Optional[float] = None,
    use_spline: bool = True,
    backend: Backend = Backend.python,
) -> float:
    r"""Compute the triangulation
    which is the last intersection of the
    :math:`\mathrm{APD} \; p_{\mathrm{high}}`
    line minus the last intersection of the
    :math:`\mathrm{APD} \; p_{\mathrm{low}}`
    line

    Parameters
    ----------
    V : Array
        The signal
    t : Array
        The time stamps
    low : int, optional
        Lower APD value, by default 30
    high : int, optional
        Higher APD value, by default 80
    v_r : Optional[float], optional
        Resting value, by default None. Only applicablle for python Backend.
    use_spline : bool, optional
        Use spline interpolation, by default True.
        Only applicable for python Backend.
    backend : utils.Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'



    Returns
    -------
    float
        The triangulation
    """
    apd_low_point = apd_point(
        factor=low,
        V=V,
        t=t,
        v_r=v_r,
        use_spline=use_spline,
    )
    apd_high_point = apd_point(
        factor=high,
        V=V,
        t=t,
        v_r=v_r,
        use_spline=use_spline,
    )
    tri = apd_high_point[-1] - apd_low_point[-1]
    if tri < 0:
        tri = np.nan
    return tri


def apd(
    factor: float,
    V: Array,
    t: Array,
    v_r: Optional[float] = None,
    use_spline: bool = True,
    backend: Backend = Backend.python,
) -> float:
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
        Only applicable for python Backend.
    backend : utils.Backend, optional
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

    if isinstance(factor, float) and 0 < factor < 1:
        # Factor should be multiplied with 100
        factor = int(factor * 100)
    _check_factor(factor)

    y = np.array(V)
    x = np.array(t)
    if y.shape != x.shape:
        raise ValueError("The signal and time are not of same lenght")

    if backend == Backend.python:
        try:
            x1, x2 = apd_point(factor=factor, V=y, t=x, v_r=v_r, use_spline=use_spline)
        except RuntimeError:
            # Return a number that indicate that something went wrong
            return -1
        return x2 - x1
    elif backend == Backend.c:
        return _c.apd(y=y, t=x, factor=factor)
    else:
        return _numba.apd(V=y, T=x, factor=factor)


def apd_point(
    factor: float,
    V: Array,
    t: Array,
    v_r: Optional[float] = None,
    use_spline=True,
) -> Tuple[float, float]:
    """Return the first and second intersection
    of the APD p line

    Parameters
    ----------
    factor : int
        The APD line
    V : np.ndarray
        The signal
    t : np.ndarray
        The time stamps
    v_r : Optional[float], optional
        The resting value, by default None
    use_spline : bool, optional
        Use spline iterpolation or not, by default True

    Returns
    -------
    Tuple[float, float]
        Two poits corresonding to the first and second
        intersection of the APD p line

    Raises
    ------
    RuntimeError
        If spline interpolation failes
    """

    _check_factor(factor)
    y = utils.normalize_signal(V, v_r) - (1 - factor / 100)

    if use_spline:

        try:
            f = UnivariateSpline(t, y, s=0, k=3)
        except Exception as ex:
            msg = (
                f"Unable to compute APD {factor * 100} using spline interpolation. "
                f"Please change your settings, {ex}"
            )
            logger.warning(msg)
            raise RuntimeError(msg)

        inds = f.roots()
        if len(inds) == 1:
            # Safety guard for strange interpolations
            inds = t[np.where(np.diff(np.sign(y)))[0]]
    else:
        inds = t[np.where(np.diff(np.sign(y)))[0]]
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
    factor: float,
    V: Array,
    t: Array,
    v_r: Optional[float] = None,
    use_spline=True,
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
    x1, x2 = apd_point(factor=factor, V=y, t=x, v_r=v_r, use_spline=use_spline)
    g = UnivariateSpline(x, y, s=0)
    y1 = g(x1)
    y2 = g(x2)

    yth = np.min(y) + (1 - factor / 100) * (np.max(y) - np.min(y))

    return APDCoords(x1, x2, y1, y2, yth)


def tau(
    x: Array,
    y: Array,
    a: float = 0.75,
    backend: Backend = Backend.python,
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
    backend : utils.Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float
        Decay time

    """
    if backend != Backend.python:
        logger.warning("Method currently only implemented for python backend")

    Y = UnivariateSpline(x, utils.normalize_signal(y) - a, s=0, k=3)
    t_max = x[int(np.argmax(y))]
    r = Y.roots()
    if len(r) >= 2:
        t_a = r[1]
    elif len(r) == 1:
        logger.warning(
            (
                "Only one zero was found when computing tau{}. " "Result might be wrong"
            ).format(int(a * 100)),
        )
        t_a = r[0]
    else:
        logger.warning(
            (
                "No zero found when computing tau{}. "
                "Return the value of time to peak"
            ).format(int(a * 100)),
        )
        t_a = x[0]

    return t_a - t_max


def time_to_peak(
    y: Array,
    x: Array,
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
    y : Array
        The signal
    x : Array
        The time stamps
    pacing : Optional[Array], optional
        The pacing amplitude, by default None
    backend : utils.Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float
        Time to peak

    """
    if backend != Backend.python:
        logger.warning("Method currently only implemented for python backend")

    # Check some edge cases
    if len(x) != len(y):
        raise UnequalLengthError("x and y does not have the same length")
    if len(x) == 0:
        return 0
    if len(y) == 0:
        return 0

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
    x: Array,
    y: Array,
    a: float = 0.8,
    backend: Backend = Backend.python,
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
    backend : utils.Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float
        The upstroke value

    Raises
    ------
    ValueError
        If a is outside the range of (0, 1)
    """
    if backend != Backend.python:
        logger.warning("Method currently only implemented for python backend")

    if not 0 < a < 1:
        raise ValueError("'a' has to be between 0.0 and 1.0")

    Y = UnivariateSpline(x, utils.normalize_signal(y) - (1 - a), s=0, k=3)
    t_max = x[int(np.argmax(y))]
    r = Y.roots()
    if len(r) >= 1:
        if len(r) == 1:
            logger.warning(
                (
                    "Only one zero was found when computing upstroke{}. "
                    "Result might be wrong"
                ).format(int(a * 100)),
            )
        t_a = r[0]
    else:
        logger.warning(
            (
                "No zero found when computing upstroke{}. "
                "Return the value of time to peak"
            ).format(int(a * 100)),
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
    signals: List[Array],
    times: List[Array],
    unit: str = "ms",
) -> Sequence[float]:
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
    List[float]
        Beating frequency in Hz for each beat
    """

    t_maxs = [t[int(np.argmax(c))] for c, t in zip(signals, times)]

    dt = np.diff(t_maxs)
    if unit == "ms":
        dt = np.divide(dt, 1000.0)
    return np.divide(1, dt)


def find_upstroke_values(
    t: np.ndarray,
    y: np.ndarray,
    upstroke_duration: int = 50,
    normalize: bool = True,
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


def apd_up_xy(
    y: Array,
    t: Array,
    factor_x: int,
    factor_y: int,
    backend: Backend = Backend.python,
) -> float:
    """Find the duration between first intersection (i.e
    during the upstroke) of two APD lines

    Arguments
    ---------
    t : np.ndarray
        Time values
    y : np.ndarray
        The trace
    from_APD: int
        First APD line (value between 0 and 100)
    to_APD: int
        Second APD line (value between 0 and 100)
    backend : utils.Backend, optional
        Which backend to use by default Backend.python.
        Choices, 'python', 'c', 'numba'

    Returns
    -------
    float:
        The time between `factor_x` to `factor_y`



    """
    _check_factor(factor_x)
    _check_factor(factor_y)

    y = numpyfy(y)
    t = numpyfy(t)

    if backend == Backend.c:
        return _c.apd_up_xy(y=y, t=t, factor_x=factor_x, factor_y=factor_y)
    if backend == Backend.numba:
        return _numba.apd_up_xy(y=y, t=t, factor_x=factor_x, factor_y=factor_y)

    y_norm = utils.normalize_signal(y)

    y_from = UnivariateSpline(t, y_norm - factor_x / 100, s=0, k=3)
    t_from = y_from.roots()[0]

    y_to = UnivariateSpline(t, y_norm - factor_y / 100, s=0, k=3)
    t_to = y_to.roots()[0]

    return t_to - t_from


def max_relative_upstroke_velocity(
    t: np.ndarray,
    y: np.ndarray,
    upstroke_duration: int = 50,
    sigmoid_fit: bool = True,
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
    maximum derivate of the sigmoid
    else:
    5. Compute the successive differences in amplitude (i.e delta y)
    and report the maximum value of these


    """

    # Interpolate to 1ms precision
    t0, y0 = utils.interpolate(t, y, dt=1.0)
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
            time_APD20_to_APD80 = apd_up_xy(s, t_upstroke, 20, 80)

        else:
            # Find max upstroke
            index = np.argmax(np.diff(upstroke))  # type:ignore
            value = np.max(np.diff(upstroke))
            x0 = None
            s = None
            time_APD20_to_APD80 = apd_up_xy(upstroke, t_upstroke, 20, 80)

    return Upstroke(
        index=index,
        value=value,
        upstroke=upstroke,
        dt=dt,
        x0=x0,
        sigmoid=s,
        time_APD20_to_APD80=time_APD20_to_APD80,
    )


def maximum_upstroke_velocity(
    y: Array,
    t: Optional[Array] = None,
    use_spline: bool = True,
    normalize: bool = False,
) -> float:
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
        y = utils.normalize_signal(y)

    if use_spline:
        f = UnivariateSpline(t, y, s=0, k=5)
        h = f.derivative()
        max_upstroke_vel = np.max(h(h.derivative().roots()))
    else:
        max_upstroke_vel = np.max(np.divide(np.diff(y), np.diff(t)))

    return max_upstroke_vel


def integrate_apd(y, t=None, factor=30, use_spline=True, normalize=False):
    r"""
    Compute the integral of the signals above
    the APD p line

    Arguments
    ---------
    y : array
        The signal
    t : array
        The time points
    factor: float
        Which APD line, by default 0.3
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
        y = utils.normalize_signal(y)

    if t is None:
        t = range(len(y))

    if isinstance(factor, float) and 0 < factor < 1:
        # Factor should be multiplied with 100
        factor = int(factor * 100)
    _check_factor(factor)

    x1, x2 = apd_point(factor, y, t, use_spline=use_spline)

    g = UnivariateSpline(t, y, s=0, k=3)

    if use_spline:
        Y = y - g(x1)
        f = UnivariateSpline(t, Y, s=0, k=3)
        integral = f.integral(x1, x2)

    else:
        val_th = np.min(y) + (1 - factor / 100) * (np.max(y) - np.min(y))
        Y = y - val_th

        t1 = t.tolist().index(x1)
        t2 = t.tolist().index(x2) + 1

        integral = np.sum(np.multiply(Y[t1:t2], np.diff(t)[t1:t2]))

    return integral


def corrected_apd(apd: float, beat_rate: float, formula: str = "friderica"):
    """Correct the given APD (or any QT measurement) for the beat rate.
    normally the faster the HR (or the shorter the RR interval),
    the shorter the QT interval, and vice versa

    Parameters
    ----------
    apd: float
        The action potential duration
    beat_rate : float
        The beat rate (number of beats per minute)
    formule : str, optional
        Formule for computing th corrected APD, either
        'friderica' or 'bazett', by default 'friderica',

    Returns
    -------
    float
        The corrected APD

    Notes
    -----

    Friderica formula (default):

    .. math::

        APD (RR)^{-1/3}

    Bazett formula:

    .. math::

        APD (RR)^{-1/2}

    where :math:`RR` is the R-R interaval in an ECG. For an action potential
    this would be equivalent to the inverse of the beating frequency (or 60
    divided by the beat rate)

    .. rubric::
        Luo, Shen, et al. "A comparison of commonly used QT correction formulae:
        the effect of heart rate on the QTc of normal ECGs." Journal of
        electrocardiology 37 (2004): 81-90.

    """

    formula = formula.lower()
    formulas = ["friderica", "bazett"]
    msg = f"Expected formula to be one of {formulas}, got {formula}"
    assert formula in formulas, msg
    RR = np.divide(60, beat_rate)

    if formula == "friderica":
        return np.multiply(apd, pow(RR, -1 / 3))
    else:
        return np.multiply(apd, pow(RR, -1 / 2))


def detect_ead(
    y: Array,
    sigma: float = 1,
    prominence_level: float = 0.07,
) -> Tuple[bool, Optional[int]]:
    """Detect (Early afterdepolarizations) EADs
    based on peak prominence.

    Parameters
    ----------
    y : Array
        The signal that you want to detect EADs
    sigma : float
        Standard deviation in the gaussian smoothing kernal
        Default: 1.0
    prominence_level: float
        How prominent a peak should be in order to be
        characterized as an EAD. This value shold be
        between 0 and 1, with a greater value being
        more prominent. Defaulta: 0.07

    Returns
    -------
    bool:
        Flag indicating if an EAD is found or not
    int or None:
        Index where we found the EAD. If no EAD is found then
        this will be None. I more than one peaks are found then
        only the first will be returned.

    Notes
    -----
    Given a signal :math:`y` we want to determine wether we have
    an EAD present in the signal. `EADs <https://en.wikipedia.org/wiki/Afterdepolarization>`_
    are abnormal depolarizations happening after the upstroke in an action potential.

    We assume that an EAD occurs betweeen the maximum value of the signal
    (i.e the peak) and the next minimum value (i.e when the signal is at rest)

    To remove noisy patterns we first smooth the signal
    with a gaussian filter. Then we take out only the part
    of the signal that is between its maximum and the next
    minimum values. Then we find the peaks with a
    `Topographic Prominence <https://en.wikipedia.org/wiki/Topographic_prominence>`_
    greather than the given prominence level

    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    y = np.array(y)
    idx_max = int(np.argmax(y))
    idx_min = idx_max + int(np.argmin(y[idx_max:]))

    y_tmp = y[idx_max:idx_min] - y[idx_min]
    if len(y_tmp) == 0:
        return False, None

    y_smooth = gaussian_filter1d(y_tmp / np.max(y_tmp), sigma)
    peaks, props = find_peaks(y_smooth, prominence=prominence_level)

    return len(peaks) > 0, None if len(peaks) == 0 else int(peaks[0] + idx_max)


def cost_terms_trace(y: Array, t: Array, backend: Backend = Backend.c) -> np.ndarray:
    y = numpyfy(y)
    t = numpyfy(t)

    if backend == Backend.python:
        logger.warning(
            "Method currently not implemented for python backend (and will probably not be)",
        )

    if backend == Backend.numba:
        return _numba.cost_terms_trace(y=y, t=t)

    # Use C backend
    return _c.cost_terms_trace(y=y, t=t)


def cost_terms(
    v: Array,
    ca: Array,
    t_v: Array,
    t_ca: Array,
    backend: Backend = Backend.c,
) -> np.ndarray:
    v = numpyfy(v)
    t_v = numpyfy(t_v)
    ca = numpyfy(ca)
    t_ca = numpyfy(t_ca)

    if backend == Backend.python:
        logger.warning(
            "Method currently not implemented for python backend (and will probably not be)",
        )

    if backend == Backend.numba:
        return _numba.cost_terms(v=v, ca=ca, t_v=t_v, t_ca=t_ca)

    # Use C backend
    return _c.cost_terms(v=v, ca=ca, t_v=t_v, t_ca=t_ca)


def all_cost_terms(
    arr: np.ndarray,
    t: np.ndarray,
    mask: Optional[np.ndarray] = None,
    backend: Backend = Backend.c,
    normalize_time: bool = True,
) -> np.ndarray:

    arr = numpyfy(arr)
    t = numpyfy(t)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected 'arr' to be of type numpy.ndarray got {type(arr)}")
    if not isinstance(t, np.ndarray):
        raise TypeError(f"Expected 't' to be of type numpy.ndarray got {type(t)}")
    if t.shape[0] != arr.shape[0]:
        raise ValueError(
            "Shape of 't'({t.shape}) and 'arr'({arr.shape}) does not match",
        )
    if normalize_time:
        t = t - t[0]

    if backend == Backend.python:
        logger.warning(
            "Method currently not implemented for python backend (and will probably not be)",
        )
    if backend == Backend.numba:
        return _numba.all_cost_terms(arr=arr, t=t, mask=mask)

    # Use C backend
    return _c.all_cost_terms(arr=arr, t=t, mask=mask)
