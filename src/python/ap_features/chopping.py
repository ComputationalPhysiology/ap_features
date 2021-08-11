import logging
from collections import namedtuple
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import utils
from .utils import Array

logger = logging.getLogger(__name__)

ChoppedData = namedtuple("ChoppedData", "data, times, pacing, parameters")
ChoppingParameters = namedtuple("ChoppingParameters", "use_pacing_info")


def chop_data(data: Array, time: Array, **kwargs) -> ChoppedData:
    """Chop data into individual beats

    Parameters
    ----------
    data : Array
        The signal amplitude
    time : Array
        Time stamps

    Returns
    -------
    ChoppedData
        Data chopped into individual beats

    """

    if time is None:
        time = np.arange(len(data))

    pacing = kwargs.pop("pacing", np.zeros(len(time)))

    if all(pacing == 0):
        logger.debug("Chop data without pacing")
        return chop_data_without_pacing(data, time, **kwargs)
    else:
        logger.debug("Chop data with pacing")
        return chop_data_with_pacing(data, time, pacing, **kwargs)


class EmptyChoppingError(RuntimeError):
    pass


class InvalidChoppingError(RuntimeError):
    pass


def chop_data_without_pacing(
    data: Array,
    time: Array,
    threshold_factor: float = 0.3,
    min_window: float = 50,
    max_window: float = 2000,
    N: Optional[int] = None,
    extend_front: Optional[float] = None,
    extend_end: Optional[float] = None,
    **kwargs,
) -> ChoppedData:

    r"""
    Chop data into beats

    Arguments
    ---------
    data : list or array
        The data that you want to chop
    time : list or array
        The time points. If none is provided time will be
        just the indices.
    threshold_factor : float
        Thresholds for where the signal should be chopped (Default: 0.3)
    extend_front : scalar
        Extend the start of each subsignal this many milliseconds
        before the threshold is detected. Default: 300 ms
    extend_end : scalar
        Extend the end of each subsignal this many milliseconds.
        Default 60 ms.
    min_window : scalar
        Length of minimum window
    max_window : scalar
        Length of maximum window
    N : int
        Length of output signals

    Returns
    -------
    ChoppedData
        The chopped data

    Notes
    -----

    The signals extracted from the MPS data consist of data from several beats,
    and in order to e.g compute properties from an average beat, we need a way
    to chop the signal into different beats. Suppose we have the
    signal :math:`y(t),
    t \in [0, T]`, where we assume that filtering and background correction
    have allready been applied. Suppose we have :math:`N` sub-signals,
    :math:`z_1, z_2, \cdots, z_N`, each representing one beat. Let
    :math:`\tau_i = [\tau_i^0, \tau_i^1], i = 1, \cdots N` be
    non-empty intervals corresponding to the support of each sub-signal
    ( :math:`\tau_i = \{ t \in [0,T]: z_i(t) \neq 0 \}`), with
    :math:`\tau_i^j < \tau_{i+1}^j \forall i` and :math:`\bigcup_{i=1}^N
    \tau_i = [0, T]`. Note that the intersection :math:` \tau_i \cap
    \tau_{i+1}` can be non-empty. The aim is now to find good candidates
    for :math:`\tau_i^j , i = 1 \cdots N, j = 0,1`. We have two different
    scenarios, namely with or without pacing information.

    If pacing information is available we can e.g set :math:`\tau_i^0`
    to be 30 ms before each stimulus is applied and :math:`\tau_i^1`
    to be 60 ms before the next stimulus is applied.

    If pacing information is not available then we need to estimate the
    beginning and end of each interval. We proceed as follows:


    1.  Choose some threshold value :math:`\eta` (default :math:`\eta = 0.5`),
        and compute :math:`y_{\eta}`

    2. Let :math:`h = y - y_{\eta}`,  and define the set

    .. math::
        \mathcal{T}_0 = \{ t : h(t) = 0 \}

    3. Sort the elements of :math:`\mathcal{T}_0` in increasing order,
       i.e :math:`\mathcal{T}_0 = (t_1, t_2, \cdots, t_M)`,
       with :math:`t_i < t_{i+1}`.

    4. Select a minimum duration of a beat :math:`\nu` (default
       :math:`\nu = 50` ms) and define the set

    .. math::
        \mathcal{T}_1 = \{t_i \in \mathcal{T}_0 : t_{i+1} - t_i > \eta \}

    to be be the subset of :math:`\mathcal{T}_0` where the difference between
    to subsequent time stamps are greater than :math:`\eta`.

    5. Let :math:`\delta > 0` (default :math:`\delta` = 0.1 ms), and set

    .. math::
        \{\tilde{\tau_1}^0, \tilde{\tau_2}^0, \cdots,\tilde{\tau_N}^0\}
        := \{ t \in  \mathcal{T}_1 : h(t + \delta) > 0 \} \\
        \{\tilde{\tau_1}^1, \tilde{\tau_2}^1, \cdots,\tilde{\tau_N}^1\}
        := \{ t \in  \mathcal{T}_1 : h(t + \delta) < 0 \} \\

    Note that we need to also make sure that all beats have a start and an end.

    6. Extend each subsignal at the beginning and end, i,e
       if :math:`a,b \geq 0`, define

    .. math::
        \tau_i^0 = \tilde{\tau_i}^0 - a \\
        \tau_i^1 = \tilde{\tau_i}^1 + b
    """
    logger.debug("Chopping without pacing")

    chop_pars = ChoppingParameters(
        use_pacing_info=False,
    )
    logger.debug(f"Use chopping parameters: {chop_pars}")

    try:
        starts, ends, zeros = find_start_and_ends(
            time=time,
            data=data,
            threshold_factor=threshold_factor,
            min_window=min_window,
            extend_front=extend_front,
            extend_end=extend_end,
        )
    except EmptyChoppingError:
        return ChoppedData(data=[], times=[], pacing=[], parameters=chop_pars)

    if len(zeros) <= 3:
        ## Just return the original data
        return ChoppedData(
            data=[data],
            times=[time],
            pacing=[np.zeros_like(data)],
            parameters=chop_pars,
        )

    # Storage
    chopped_data, chopped_times, chopped_pacing = chop_from_start_ends(
        data,
        time,
        starts,
        ends,
        pacing=None,
        N=N,
        max_window=max_window,
        min_window=min_window,
    )

    return ChoppedData(
        data=chopped_data,
        times=chopped_times,
        pacing=chopped_pacing,
        parameters=chop_pars,
    )


def chop_from_start_ends(
    data: Array,
    time: Array,
    starts: Array,
    ends: Array,
    pacing: Optional[Array] = None,
    N: Optional[int] = None,
    max_window: float = 2000,
    min_window: float = 50,
) -> Tuple[List[Array], List[Array], List[Array]]:
    """Chop the data based on starts and ends

    Parameters
    ----------
    data : Array
        The signal amplitude
    time : Array
        The time stamps
    starts : Array
        List of start points
    ends : Array
        List of end points
    pacing : Optional[Array], optional
        Pacing amplitude, by default None
    N : Optional[int], optional
        Number of points in each chopped signal, by default None.
        If this is differnt from None then each signal will be
        interpolated so that it has N points
    max_window : float, optional
        Maximum allowed size of a chopped signal, by default 2000
    min_window : float, optional
        Minimum allowed size of a chopped signal, by default 50

    Returns
    -------
    Tuple[List[Array], List[Array], List[Array]]
        Chopped data, times, pacing
    """

    chopped_data: List[Array] = []
    chopped_pacing: List[Array] = []
    chopped_times: List[Array] = []

    if pacing is None:
        pacing = np.zeros_like(time)

    # Make a spline interpolation
    data_spline = UnivariateSpline(time, data, s=0)
    pacing_spline = UnivariateSpline(time, pacing, s=0)

    # Add a little bit of slack
    eps = 1e-10

    for s, e in zip(starts, ends):

        if N is None:
            # Find the correct time points
            s_idx = next(i for i, si in enumerate(time) if si >= s - eps)
            e_idx = next(i for i, ei in enumerate(time) if ei >= e - eps)
            t = time[s_idx - 1 : e_idx + 1]

        else:
            t = np.linspace(s, e, N)

        if len(t) == 0:
            continue

        if t[-1] - t[0] < min_window:
            # Skip this one
            continue

        if t[-1] - t[0] > max_window:

            t_end = next(i for i, ti in enumerate(t) if ti - t[0] > max_window) + 1
            t = t[:t_end]

        chopped_data.append(data_spline(t))
        chopped_pacing.append(pacing_spline(t))
        chopped_times.append(t)
    return chopped_data, chopped_times, chopped_pacing


def find_start_and_ends(
    time: Array,
    data: Array,
    threshold_factor: float,
    min_window: float,
    extend_front: Optional[float],
    extend_end: Optional[float],
) -> Tuple[Array, Array, Array]:
    """Find the starts and ends of a signal.

    Parameters
    ----------
    time : Array
        Array of time stamps
    data : Array
        Signal amplitude
    threshold_factor : float
        Threshold for where to chop
    min_window : float
        Lenght of minimum chopped signal
    extend_front : Optional[float]
        How many ms the signal should be extended at the front
    extend_end : Optional[float]
        How many ms the signal should be extended at the end

    Returns
    -------
    Tuple[Array, Array, Array]
        starts, ends, zeros

    Raises
    ------
    EmptyChoppingError
        If starts or ends is or become zero
    """

    starts, ends, zeros = locate_chop_points(
        time,
        data,
        threshold_factor,
        min_window=min_window,
    )

    if len(zeros) <= 3:
        return starts, ends, zeros

    starts, ends = filter_start_ends_in_chopping(starts, ends, extend_front, extend_end)

    while len(ends) > 0 and ends[-1] > time[-1]:
        ends = ends[:-1]
        starts = starts[:-1]

    if len(ends) == 0:
        raise EmptyChoppingError

    # Update the ends to be the start of the next trace
    for i, s in enumerate(starts[1:]):
        ends[i] = s  # type: ignore
    ends[-1] = min(ends[-1], time[-2])  # type: ignore

    return starts, ends, zeros


def filter_start_ends_in_chopping(
    starts: Array,
    ends: Array,
    extend_front: Optional[float] = None,
    extend_end: Optional[float] = None,
) -> Tuple[Array, Array]:
    """Adjust start and ends based on extend_front
    and extent_end

    Parameters
    ----------
    starts : Array
        The start points
    ends : Array
        The end points
    extend_front : Optional[float], optional
        How much you want to extent the front. If not provided
        it will try to use the half distance between the minimum
        start and end, by default None
    extend_end : Optional[float], optional
        How much you want to extend the end. If not provided
        it will try to use the half distance between the minimum
        start and end, by default None

    Returns
    -------
    Tuple[Array, Array]
        start, ends

    Raises
    ------
    EmptyChoppingError
        If number of starts or ends become zero.
    InvalidChoppingError
        If number of starts and ends does not add up.
    """
    starts = np.array(starts)
    ends = np.array(ends)

    starts, ends = check_starts_ends(starts, ends)

    # Find the length half way between the previous and next point
    extend_front = get_extend_value(extend_front, starts, ends, default=300)
    extend_end = get_extend_value(extend_end, starts, ends, default=60)

    # Subtract the extend front
    starts = np.subtract(starts, extend_front)

    # Add at the end
    ends = np.add(ends, extend_end)

    # If the trace starts in the middle of an event, that event is thrown out
    if starts[0] < 0:
        starts = starts[1:]

    new_starts = []
    new_ends = []
    for s in np.sort(starts):
        # if len(new_ends) !
        new_starts.append(s)
        for e in np.sort(ends):
            if e > s:
                new_ends.append(e)
                break
        else:
            # If no end was appended
            # we pop the last start
            new_starts.pop()

    return np.array(new_starts), np.array(new_ends)


def get_extend_value(
    extend: Optional[float],
    starts: Array,
    ends: Array,
    default: float,
) -> float:
    if extend is None:
        try:
            value = float(np.min(np.subtract(starts[1:], ends[:-1])) / 2)
        except IndexError:
            value = default
    else:
        value = extend
    return value


def check_starts_ends(starts: Array, ends: Array) -> Tuple[Array, Array]:
    """Check starts and ends and make sure that
    they are consitent

    Parameters
    ----------
    starts : Array
        List of start points
    ends : Array
        List of end points

    Returns
    -------
    Tuple[Array, Array]
        starts, ends

    Raises
    ------
    EmptyChoppingError
        If number of starts or ends become zero.
    InvalidChoppingError
        If number of starts and ends does not add up.
    """
    # If there is no starts return nothing
    if len(starts) == 0:
        raise EmptyChoppingError

    # The same with no ends
    if len(ends) == 0:
        raise EmptyChoppingError

    # If the first end is lower than the first start
    # we should drop the first end
    try:
        while ends[0] < starts[0]:
            ends = ends[1:]
    except IndexError as ex:
        raise EmptyChoppingError from ex

    # If we have a beat at the end without an end we remove
    # that one
    try:
        while starts[-1] > ends[-1]:
            starts = starts[:-1]
    except IndexError as ex:
        raise EmptyChoppingError from ex

    # And we should check this one more time
    if len(ends) == 0:
        raise EmptyChoppingError

    if len(ends) != len(starts):
        raise InvalidChoppingError(
            f"Unequal number of starts {len(starts)} and ends {len(ends)}",
        )
    return starts, ends


def locate_chop_points(
    time: Array,
    data: Array,
    threshold_factor: float,
    min_window: float = 50,
    eps: float = 0.1,
) -> Tuple[Array, Array, Array]:
    """Find the ponts where to chop

    Parameters
    ----------
    time : Array
        Time stamps
    data : Array
        The signal amplitide
    threshold_factor : float
        The thresold for where to chop
    min_window : float, optional
        Mininmum allow size of signal in ms, by default 50
    eps : float, optional
        Perterbation use to find the sign of the signal
        derivative, by default 0.1

    Returns
    -------
    Tuple[Array, Array, Array]
        starts, ends, zeros
    """

    # Data with zeros at the threshold
    data_spline_thresh = UnivariateSpline(
        time,
        utils.normalize_signal(data) - threshold_factor,
        s=0,
    )
    # Localization of the zeros
    zeros_threshold_ = data_spline_thresh.roots()

    # Remove close indices
    inds = min_window < np.diff(zeros_threshold_) * 2
    zeros_threshold = np.append(zeros_threshold_[0], zeros_threshold_[1:][inds])

    # Find the starts
    starts = zeros_threshold[data_spline_thresh(zeros_threshold + eps) > 0]

    # Find the endpoint where we hit the threshold
    ends = zeros_threshold[data_spline_thresh(zeros_threshold + eps) < 0]

    return starts, ends, zeros_threshold


def chop_data_with_pacing(
    data: Array,
    time: Array,
    pacing: Array,
    extend_front: float = 0,
    extend_end: float = 0,
    min_window: float = 300,
    max_window: float = 2000,
    **kwargs,
) -> ChoppedData:
    """
    Chop data based on pacing

    Arguments
    ---------
    data : Array
        The data to be chopped
    time : Array
        The time stamps for the data
    pacing : Array
        The pacing amplitude
    extend_front : float
        Extend the start of each subsignal this many milliseconds
        before the threshold is detected. Default: 0 ms
    extend_end : float
        Extend the end of each subsignal this many milliseconds.
        Default 0 ms.
    min_window : float
        Minimum size of chopped signal
    max_window : float
        Minimum size of chopped signal

    Returns
    -------
    ChoppeData
        Named tuple with the choppped data


    Notes
    -----
    We first find where the pacing for each beat starts by finding
    the indices where the pacing amplitude goes form zero to
    something positive. Then we iterate over these indices and let
    the start of each beat be the start of the pacing (minus `extend_front`,
    which has a default value of zero) and the end will be the time of the
    beginning of the next beat (plus `extend_end` which is also has as
    default value of zero).
    """

    # Find indices for start of each pacing
    starts, ends = pacing_to_start_ends(time, pacing, extend_front, extend_end)

    chopped_data, chopped_times, chopped_pacing = chop_from_start_ends(
        data,
        time,
        starts,
        ends,
        pacing=pacing,
        N=None,
        max_window=max_window,
        min_window=min_window,
    )

    chop_pars = ChoppingParameters(use_pacing_info=True)

    return ChoppedData(
        data=chopped_data,
        times=chopped_times,
        pacing=chopped_pacing,
        parameters=chop_pars,
    )


def find_pacing_indices(pacing: Array) -> np.ndarray:
    """Return indices of where the pacing array changes
    from a low to a high values

    Parameters
    ----------
    pacing : list or np.ndarray
        The array with pacing amplitude

    Returns
    -------
    np.ndarray
        List of indices where the pacing is triggered
    """
    return np.where(np.diff(np.array(pacing, dtype=float)) > 0)[0].tolist()


def pacing_to_start_ends(
    time: Array,
    pacing: Array,
    extend_front: float,
    extend_end: float,
    add_final: bool = True,
) -> Tuple[Array, Array]:
    """Convert an array of pacing amplitudes to
    start and end points

    Parameters
    ----------
    time : Array
        Time stamps
    pacing : Array
        Array of pacing amplitides
    extend_front : float
        How many ms you want to extend the array in front
    extend_end : float
        How many ms you want to exten the array at the end
    add_final : bool, optional
        Wheter you want to add a final end point for the
        last index, by default True

    Returns
    -------
    Tuple[Array, Array]
        starts, ends
    """
    start_pace_idx = np.where(np.diff(np.array(pacing, dtype=float)) > 0)[0].tolist()
    N = len(time)
    if add_final:
        start_pace_idx.append(N - 1)

    indices = np.array(start_pace_idx)
    time = np.array(time)
    starts = time[indices[:-1]]
    ends = time[indices[1:]]

    return filter_start_ends_in_chopping(starts, ends, extend_front, extend_end)
