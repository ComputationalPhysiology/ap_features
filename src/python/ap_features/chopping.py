import logging
from collections import namedtuple
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import filters
from . import utils
from .utils import Array

logger = logging.getLogger(__name__)

ChoppedData = namedtuple(
    "ChoppedData",
    "data, times, pacing, parameters, intervals, upstroke_times",
)
Interval = namedtuple("Interval", "start, end")


def find_start_end_index(
    time_stamps: Array,
    start: float,
    end: float,
) -> Tuple[int, int]:
    start_index = 0
    end_index = len(time_stamps)
    if start > 0:
        try:
            start_index = next(i for i, v in enumerate(time_stamps) if v >= start)
        except StopIteration:
            pass

    if end != -1:
        try:
            end_index = next(i for i, v in enumerate(time_stamps) if v >= end)
        except StopIteration:
            pass
    return start_index, end_index


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

    if np.isnan(data).any():
        # If the array contains nan values we should not analyze it
        return ChoppedData(
            data=[],
            times=[],
            pacing=[],
            parameters={},
            intervals=[],
            upstroke_times=[],
        )

    pacing = kwargs.pop("pacing", np.zeros(len(time)))
    ignore_pacing = kwargs.pop("ignore_pacing", False)

    if ignore_pacing or all(pacing == 0):
        logger.debug("Chop data without pacing")
        return chop_data_without_pacing(data, time, **kwargs)
    else:
        logger.debug("Chop data with pacing")
        return chop_data_with_pacing(data, time, pacing, **kwargs)


class EmptyChoppingError(RuntimeError):
    pass


class InvalidChoppingError(RuntimeError):
    pass


def default_chopping_options():
    return dict(
        threshold_factor=0.5,
        min_window=50,
        max_window=2000,
        N=None,
        extend_front=None,
        extend_end=None,
        ignore_pacing=False,
        intervals=None,
    )


def chop_data_without_pacing(
    data: Array,
    time: Array,
    threshold_factor: float = 0.5,
    min_window: float = 50,
    max_window: float = 2000,
    N: Optional[int] = None,
    extend_front: Optional[float] = None,
    extend_end: Optional[float] = None,
    intervals: Optional[List[Interval]] = None,
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

    chop_pars = {
        "use_pacing_info": False,
        "threshold_factor": threshold_factor,
        "min_window": min_window,
        "max_window": max_window,
        "N": N,
        "extend_front": extend_front,
        "extend_end": extend_end,
        "intervals": intervals,
    }
    logger.debug(f"Use chopping parameters: {chop_pars}")

    if intervals is None:
        try:
            intervals, upstroke_times = find_start_and_ends(
                time=time,
                data=data,
                threshold_factor=threshold_factor,
                min_window=min_window,
                extend_front=extend_front,
                extend_end=extend_end,
            )
        except EmptyChoppingError:
            return ChoppedData(
                data=[],
                times=[],
                pacing=[],
                parameters=chop_pars,
                intervals=[],
                upstroke_times=[],
            )

        if len(upstroke_times) <= 2:
            ## Just return the original data
            return ChoppedData(
                data=[data],
                times=[time],
                pacing=[np.zeros_like(data)],
                parameters=chop_pars,
                intervals=intervals,
                upstroke_times=upstroke_times,
            )
    else:
        # Use the start of each interval as zero
        upstroke_times = [interval[0] for interval in intervals]

    # Storage
    chopped_data, chopped_times, chopped_pacing = chop_intervals(
        data,
        time,
        intervals,
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
        intervals=intervals,
        upstroke_times=upstroke_times,
    )


def chop_intervals(
    data: Array,
    time: Array,
    intervals: List[Interval],
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
    intervals : List[Interval]
        List of intervals with start and ends
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

    for s, e in intervals:

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


def cutoff_final_interval(intervals: List[Interval], end_time: float) -> List[Interval]:
    new_intervals = intervals.copy()
    if len(intervals) > 0:
        final_interval = new_intervals.pop()
        new_intervals.append(
            Interval(start=final_interval[0], end=min(final_interval[1], end_time)),
        )
    return new_intervals


def find_start_and_ends(
    time: Array,
    data: Array,
    threshold_factor: float,
    min_window: float,
    extend_front: Optional[float],
    extend_end: Optional[float],
) -> Tuple[List[Interval], Array]:
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
    Tuple[List[Interval], Array]
        (starts, ends) for interval, and upstroke times

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

    intervals = filter_start_ends_in_chopping(starts, ends, extend_front, extend_end)
    intervals = cutoff_final_interval(intervals, time[-1])

    return intervals, starts


def create_intervals(starts: Array, ends: Array) -> List[Interval]:
    remaing_ends = (e for e in sorted(ends))
    intervals = []
    for start in sorted(starts):
        for end in remaing_ends:
            if end > start:
                intervals.append(Interval(start, end))
                break
    return intervals


def extend_intervals(intervals, extend_front, extend_end):
    extended_intervals = []
    for interval in intervals:
        extended_intervals.append(
            Interval(
                start=max(interval[0] - extend_front, 0),
                end=interval[1] + extend_end,
            ),
        )
    return extended_intervals


def filter_start_ends_in_chopping(
    starts: Array,
    ends: Array,
    extend_front: Optional[float] = None,
    extend_end: Optional[float] = None,
) -> List[Interval]:
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
    List[Interval]
        List of the filtered intervals

    Raises
    ------
    EmptyChoppingError
        If number of starts or ends become zero.
    InvalidChoppingError
        If any of the intervals have a start value that
        is higher than the end value.
    """
    starts = np.array(starts)
    ends = np.array(ends)

    intervals = create_intervals(starts, ends)

    check_intervals(intervals)

    # Find the length half way between the previous and next point
    extend_front = get_extend_value(extend_front, intervals, default=300)
    extend_end = get_extend_value(extend_end, intervals, default=60)

    intervals = extend_intervals(intervals, extend_front, extend_end)

    check_intervals(intervals)

    return intervals


def get_extend_value(
    extend: Optional[float],
    intervals: List[Interval],
    default: float,
) -> float:
    starts = [interval[0] for interval in intervals]
    ends = [interval[1] for interval in intervals]
    if extend is None:
        try:
            value = max(float(np.min(np.subtract(starts[1:], ends[:-1])) / 2), 0)
        except (IndexError, ValueError):
            value = default
    else:
        value = extend
    return value


def check_intervals(intervals: List[Interval]) -> None:
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
    if len(intervals) == 0:
        raise EmptyChoppingError("No intervals found")

    for interval in intervals:
        if interval[0] >= interval[1]:
            raise InvalidChoppingError(f"Interval {interval} is not allowed")


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
        utils.normalize_signal(filters.filt(data)) - threshold_factor,
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
    intervals: Optional[List[Interval]] = None,
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

    if intervals is None:
        # Find indices for start of each pacing
        intervals, upstroke_times = pacing_to_start_ends(
            time,
            pacing,
            extend_front,
            extend_end,
        )
    else:
        upstroke_times = [interval[0] for interval in intervals]

    chopped_data, chopped_times, chopped_pacing = chop_intervals(
        data,
        time,
        intervals,
        pacing=pacing,
        N=None,
        max_window=max_window,
        min_window=min_window,
    )

    chop_pars = {"use_pacing_info": True}

    return ChoppedData(
        data=chopped_data,
        times=chopped_times,
        pacing=chopped_pacing,
        parameters=chop_pars,
        intervals=intervals,
        upstroke_times=upstroke_times,
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
) -> Tuple[List[Interval], Array]:
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
    Tuple[List[Interval], Array]
        (starts, ends) for interval, and upstroke times

    """
    start_pace_idx = np.where(np.diff(np.array(pacing, dtype=float)) > 0)[0].tolist()
    N = len(time)
    if add_final:
        start_pace_idx.append(N - 1)

    indices = np.array(start_pace_idx)
    time = np.array(time)
    starts = time[indices[:-1]]
    ends = time[indices[1:]]

    intervals = filter_start_ends_in_chopping(starts, ends, extend_front, extend_end)
    return intervals, starts
