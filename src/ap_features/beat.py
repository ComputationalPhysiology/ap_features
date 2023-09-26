from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import average
from . import background
from . import chopping
from . import features
from . import filters as _filters
from . import plot
from . import utils
from .background import BackgroundCorrection as BC
from .utils import Array
from .utils import Backend


def identity(x: Any, y: Any = None, z: Any = None) -> Any:
    return x


def copy_function(copy: bool):
    if copy:
        return np.copy
    else:
        return identity


class Trace:
    __slots__ = ("_t", "_y", "_pacing", "_backend")

    def __init__(
        self,
        y: Array,
        t: Optional[Array],
        pacing: Optional[Array] = None,
        backend: Backend = Backend.numba,
    ) -> None:
        if t is None:
            t = np.arange(len(y))
        self._t = utils.numpyfy(t)
        self._y = utils.numpyfy(y)
        if pacing is None:
            pacing = np.zeros_like(self._t)  # type: ignore
        self._pacing = utils.numpyfy(pacing)
        self._validate_array_sizes()

        assert backend in Backend
        self._backend = backend

    def _validate_array_sizes(self):
        if len(self._y) != len(self._t):
            raise ValueError(
                f"Expected y (size={len(self._y)}) and "
                f"t (size={len(self._t)}) to have same size",
            )
        if len(self._y) != len(self._pacing):
            raise ValueError(
                f"Expected y (size={len(self._y)}) and "
                f"pacing (size={len(self._pacing)}) to have same size",
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(t={self.t.shape}, y={self.y.shape})"

    @property
    def time_unit(self) -> str:
        """The time unit 'ms' or 's'"""
        return utils.time_unit(self.t)

    @property
    def duration(self) -> float:
        """The duration of the trace, i.e last time point minus the first"""
        return self.t[-1] - self.t[0]

    def ensure_time_unit(self, unit: str) -> None:
        """Convert time to milliseconds or seconds

        Parameters
        ----------
        unit : str
            A string with 'ms' or 's'
        """
        assert unit in ["ms", "s"], f"Expected unit to be 'ms' or 's', got {unit}"

        if self.time_unit != unit:
            unitfactor = 1 / 1000.0 if unit == "s" else 1000.0
            self.t[:] *= unitfactor

    @property
    def t(self) -> np.ndarray:
        """The time stamps"""
        return self._t

    @property
    def y(self) -> np.ndarray:
        """The trace"""
        return self._y

    def max(self) -> float:
        """Return the maximum value of the trace"""
        return self.y.max()

    def min(self) -> float:
        """Return the minimum value of the trace"""
        return self.y.min()

    def amp(self) -> float:
        """Return the difference between the maximum and minimum value"""
        return self.max() - self.min()

    @property
    def pacing(self) -> np.ndarray:
        """Array of pacing amplitudes. If no pacing info is available,
        then this will be an array of zeros with same length as the trace"""
        return self._pacing

    def __len__(self):
        return len(self.y)

    def __eq__(self, other) -> bool:
        try:
            return (
                (self.t == other.t).all()
                and (self.y == other.y).all()
                and (self.pacing == other.pacing).all()
            )
        except Exception:
            return False

    def slice(self, start: float, end: float, copy: bool = True) -> "Trace":
        """Create a slice of the original trace

        Parameters
        ----------
        start : float
            Start time of slice
        end : float
            End time of slice
        copy : bool, optional
            If true create a copy of the original array otherwise return a slice
            of the original array, by default True

        Returns
        -------
        Trace
            A sliced trace
        """
        f = copy_function(copy)
        start_index, end_index = chopping.find_start_end_index(self.t, start, end)
        return self.__class__(
            y=f(self.y)[start_index:end_index],
            t=f(self.t)[start_index:end_index],
            pacing=f(self.pacing)[start_index:end_index],
            backend=self._backend,
        )

    def copy(self, **kwargs):
        """Create a copy of the trace"""
        return self.__class__(
            y=np.copy(self.y),
            t=np.copy(self.t),
            pacing=np.copy(self.pacing),
            backend=self._backend,
            **kwargs,
        )

    def plot(
        self,
        fname: str = "",
        include_pacing: bool = False,
        include_background: bool = False,
        ylabel: str = "",
    ):
        """Plot the trace with matplotlib

        Parameters
        ----------
        fname : str, optional
            Name of the figure to be saved. If not provided the
            figure will note be saved, but you can show by calling
            `plt.show`
        include_pacing : bool, optional
            Whether to include pacing in the plot, by default False
        include_background : bool, optional
            Whether to include the background, by default False
        ylabel : str, optional
            Label on the y-axis, by default ""

        Returns
        -------
        Tuple[plt.Figure, plt.Axes] | None
            If matplotlib is installed it will return a tuple containing
            the figure and the axes.
        """
        return plot.plot_beat(
            self,
            include_pacing=include_pacing,
            include_background=include_background,
            ylabel=ylabel,
            fname=fname,
        )

    def as_spline(self, k: int = 3, s: Optional[int] = None) -> UnivariateSpline:
        """Convert trace to spline

        Parameters
        ----------
        k : int, optional
            Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
            Default is `k` = 3, a cubic spline, by default 3.
        s : float or None, optional
            Positive smoothing factor used to choose the number of knots.
            If 0, spline will interpolate through all data points,
            by default 0

        Returns
        -------
        UnivariateSpline
            A spline representation of the trace
        """
        return UnivariateSpline(x=self.t, y=self.y, k=k, s=s)


class Beat(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        y_rest: Optional[float] = None,
        y_max: Optional[float] = None,
        parent: Optional["Beats"] = None,
        backend: Backend = Backend.numba,
        beat_number: Optional[int] = None,
    ) -> None:
        super().__init__(y, t, pacing=pacing, backend=backend)
        msg = (
            "Expected shape of 't' and 'y' to be the same. got "
            f"{self._t.shape}(t) and {self._y.shape}(y)"
        )
        assert self._t.shape == self._y.shape, msg
        self._y_rest = y_rest
        self._y_max = y_max
        self._parent = parent
        self._beat_number = beat_number

    @property
    def y_normalized(self) -> np.ndarray:
        """Return normalized signal"""
        return utils.normalize_signal(self.y, v_r=self.y_rest, v_max=self.y_max)

    @property
    def y_rest(self) -> Optional[float]:
        """Return resting value if specified, otherwise None"""
        return self._y_rest

    @property
    def y_max(self) -> Optional[float]:
        """Return maximum value if specified, otherwise None"""
        return self._y_max

    def as_beats(self) -> "Beats":
        """Convert trace from Beat to Beats"""
        return Beats(y=self.y, t=self.y, pacing=self.pacing)

    def is_valid(self):
        """Check if intersection with APD50 line gives two
        points
        """
        f = UnivariateSpline(self.t, self.y_normalized - 0.5, s=0, k=3)
        return f.roots().size == 2

    @property
    def parent(self) -> Optional["Beats"]:
        """If the beat comes from several Beats
        object then this will return those Beats

        Returns
        -------
        Beats
            The parent Beats
        """
        return self._parent

    def apd_point(
        self,
        factor: float,
        use_spline: bool = True,
        strategy: features.APDPointStrategy = features.APDPointStrategy.big_diff_plus_one,
    ) -> Tuple[float, float]:
        """Return the first and second intersection
        of the APD p line

        Parameters
        ----------
        factor : int
            The APD
        use_spline : bool, optional
            Use spline interpolation or not, by default True

        Returns
        -------
        Tuple[float, float]
            Two points corresponding to the first and second
            intersection of the APD p line
        """
        return features.apd_point(
            factor=factor,
            V=self.y,
            t=self.t,
            v_r=self.y_rest,
            v_max=self.y_max,
            use_spline=use_spline,
            strategy=strategy,
        )

    def apd_points(self, factor: float, use_spline: bool = True) -> List[float]:
        """Return all intersections of the APD p line

        Parameters
        ----------
        factor : int
            The APD
        use_spline : bool, optional
            Use spline interpolation or not, by default True

        Returns
        -------
        List[float]
            Two points corresponding to the first and second
            intersection of the APD p line
        """
        return features.apd_points(
            factor=factor,
            V=self.y,
            t=self.t,
            v_r=self.y_rest,
            v_max=self.y_max,
            use_spline=use_spline,
        )

    def apd(self, factor: float, use_spline: bool = True) -> float:
        """The action potential duration

        Parameters
        ----------
        factor : int
            Integer between 0 and 100
        use_spline : bool, optional
            Use spline interpolation, by default True.

        Returns
        -------
        float
            action potential duration
        """
        return features.apd(
            factor=factor,
            V=self.y,
            t=self.t,
            v_r=self.y_rest,
            v_max=self.y_max,
            use_spline=use_spline,
        )

    def triangulation(
        self,
        low: int = 30,
        high: int = 80,
        use_spline: bool = True,
    ) -> float:
        r"""Compute the triangulation
        which is the last intersection of the
        :math:`\mathrm{APD} \; p_{\mathrm{high}}`
        line minus the last intersection of the
        :math:`\mathrm{APD} \; p_{\mathrm{low}}`
        line

        Parameters
        ----------
        low : int, optional
            Lower APD value, by default 30
        high : int, optional
            Higher APD value, by default 80
        use_spline : bool, optional
            Use spline interpolation, by default True.

        Returns
        -------
        float
            The triangulations
        """
        return features.triangulation(
            V=self.y,
            t=self.t,
            low=low,
            high=high,
            v_r=self.y_rest,
            v_max=self.y_max,
            use_spline=use_spline,
        )

    def capd(
        self,
        factor: float,
        beat_rate: Optional[float] = None,
        formula: str = "friderica",
        use_spline: bool = True,
        use_median_beat_rate: bool = False,
    ) -> float:
        """Correct the given APD (or any QT measurement) for the beat rate.
        normally the faster the HR (or the shorter the RR interval),
        the shorter the QT interval, and vice versa

        Parameters
        ----------
        factor : int
            Integer between 0 and 100
        beat_rate : float
            The beat rate (number of beats per minute)
        formula : str, optional
            Formula for computing th corrected APD, either
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

        where :math:`RR` is the R-R interval in an ECG. For an action potential
        this would be equivalent to the inverse of the beating frequency (or 60
        divided by the beat rate)

        .. rubric::
            Luo, Shen, et al. "A comparison of commonly used QT correction formulae:
            the effect of heart rate on the QTc of normal ECGs." Journal of
            electrocardiology 37 (2004): 81-90.
        """

        if beat_rate is None:
            if self.parent is None:
                raise RuntimeError(
                    "Cannot compute corrected APD. Please provide beat_rate",
                )
            if self._beat_number is None or self._beat_number >= len(
                self.parent.beat_rates,
            ):
                use_median_beat_rate = True

            if use_median_beat_rate:
                beat_rate = self.parent.beat_rate
            else:
                beat_rate = self.parent.beat_rates[self._beat_number]  # type: ignore

        apd = self.apd(factor=factor, use_spline=use_spline)
        return features.corrected_apd(apd, beat_rate=beat_rate, formula=formula)

    def tau(self, a: float) -> float:
        """Decay time. Time for the signal amplitude to go from maximum to
        (1 - a) * 100 % of maximum

        Parameters
        ----------
        a : float
            The value for which you want to estimate the time decay

        Returns
        -------
        float
            Decay time
        """
        return features.tau(x=self.t, y=self.y, a=a)

    def ttp(self, use_pacing: bool = True) -> float:
        """Computed the time to peak from pacing is
        triggered to maximum amplitude. Note, if pacing
        is not provided it will compute the time from
        the beginning of the trace (which might not be consistent)
        to the peak.

        Parameters
        ----------
        use_pacing : bool, optional
            If pacing is available compute time to
            peak from start of packing, by default True

        Returns
        -------
        float
            Time to peak
        """
        pacing = self.pacing if use_pacing else None
        return features.time_to_peak(x=self.t, y=self.y, pacing=pacing)

    def time_above_apd_line(self, factor: float) -> float:
        """Compute the amount of time spent above APD p line"""
        return features.time_above_apd_line(
            V=self.y,
            t=self.t,
            factor=factor,
            v_r=self.y_rest,
            v_max=self.y_max,
        )

    def integrate_apd(
        self,
        factor: float,
        use_spline: bool = True,
        normalize: bool = False,
    ) -> float:
        """Compute the integral of the signals above
        the APD p line

        Parameters
        ----------
        factor : float
            Which APD line
        use_spline : bool, optional
            Use spline interpolation, by default True
        normalize : bool, optional
            If true normalize signal first, so that max value is 1.0,
            and min value is zero before performing the computation,
            by default False

        Returns
        -------
        float
            Integral above the APD p line
        """
        return features.integrate_apd(
            t=self.t,
            y=self.y,
            factor=factor,
            use_spline=use_spline,
            normalize=normalize,
        )

    def peaks(self, prominence_level: float = 0.1) -> List[int]:
        """Return the list of peak indices given a prominence level"""
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(self.y, prominence=prominence_level)
        return peaks

    def upstroke(self, a: float) -> float:
        """Compute the time from (1-a)*100 % signal
        amplitude to peak. For example if if a = 0.8
        if will compute the time from the starting value
        of APD80 to the upstroke.

        Parameters
        ----------
        a : float
            Fraction of signal amplitude

        Returns
        -------
        float
            The upstroke value
        """
        return features.upstroke(x=self.t, y=self.y, a=a)

    def maximum_upstroke_velocity(
        self,
        use_spline: bool = True,
        normalize: bool = False,
    ) -> float:
        """Compute maximum upstroke velocity

        Parameters
        ----------
        use_spline : bool, optional
            Use spline interpolation, by default True
        normalize : bool, optional
            If true normalize signal first, so that max value is 1.0,
            and min value is zero before performing the computation,
            by default False

        Returns
        -------
        float
            The maximum upstroke velocity
        """
        return features.maximum_upstroke_velocity(
            t=self.t,
            y=self.y,
            use_spline=use_spline,
            normalize=normalize,
        )

    def maximum_relative_upstroke_velocity(
        self,
        upstroke_duration: int = 50,
        sigmoid_fit: bool = True,
    ):
        """Estimate maximum relative upstroke velocity

        Parameters
        ----------
        upstroke_duration : int
            Duration in milliseconds of upstroke (Default: 50).
            This does not have to be exact up should at least be
            longer than the upstroke.
        sigmoid_fit : bool
            If True then use a sigmoid function to fit the data
            of the upstroke and report the maximum derivate of
            the sigmoid as the maximum upstroke.
        """
        return features.max_relative_upstroke_velocity(
            t=self.t,
            y=self.y,
            upstroke_duration=upstroke_duration,
            sigmoid_fit=sigmoid_fit,
        )

    def detect_ead(
        self,
        sigma: float = 1,
        prominence_level: float = 0.07,
    ) -> Tuple[bool, Optional[int]]:
        """Detect (Early after depolarizations) EADs
        based on peak prominence.

        Parameters
        ----------
        y : Array
            The signal that you want to detect EADs
        sigma : float
            Standard deviation in the gaussian smoothing kernel
            Default: 1.0
        prominence_level: float
            How prominent a peak should be in order to be
            characterized as an EAD. This value should be
            between 0 and 1, with a greater value being
            more prominent. Default: 0.07

        Returns
        -------
        bool:
            Flag indicating if an EAD is found or not
        int or None:
            Index where we found the EAD. If no EAD is found then
            this will be None. I more than one peaks are found then
            only the first will be returned.
        """
        return features.detect_ead(
            self.y,
            sigma=sigma,
            prominence_level=prominence_level,
        )

    def apd_up_xy(self, low, high):
        """Find the duration between first intersection (i.e
        during the upstroke) of two APD lines

        Arguments
        ---------
        low: int
            First APD line (value between 0 and 100)
        high: int
            Second APD line (value between 0 and 100)

        Returns
        -------
        float:
            The time between `low` to `high`

        """
        return features.apd_up_xy(y=self.y, t=self.t, low=low, high=high)

    @property
    def cost_terms(self):
        return features.cost_terms_trace(y=self.y, t=self.t, backend=self._backend)


def remove_bad_indices(feature_list: List[List[float]], bad_indices: Set[int]):
    if len(bad_indices) == 0:
        return feature_list

    new_list = []
    for sublist in feature_list:
        new_sublist = sublist.copy()
        for index in sorted(bad_indices, reverse=True):
            del new_sublist[index]
        new_list.append(new_sublist)
    return new_list


def align_beats(
    beats: List[Beat],
    apd_point=50,
    N=200,
    parent=None,
    strategy: features.APDPointStrategy = features.APDPointStrategy.big_diff_plus_one,
):
    if len(beats) == 0:
        return beats

    apd_points = [b.apd_point(apd_point, strategy=strategy) for b in beats]
    bad_beats = [np.isclose(*p) for p in apd_points]

    # Make sure all beats start at time zero
    xs = [beat.t - ap[0] for (beat, ap, bad) in zip(beats, apd_points, bad_beats) if not bad]
    ys = [beat.y for (beat, bad) in zip(beats, bad_beats) if not bad]

    # Make them start at zero
    min_x = np.min([np.min(xi) for xi in xs])
    xs = [xi - min_x for xi in xs]

    return [Beat(yi, xi, parent=parent) for (yi, xi) in zip(ys, xs)]


def average_beat(
    beats: List[Beat],
    N: int = 200,
    filters: Optional[Sequence[_filters.Filters]] = None,
    x: float = 1.0,
    apd_point: float = 50,
) -> Beat:
    if len(beats) == 0:
        raise ValueError("Cannot average an empty list")
    if filters is not None:
        beats = filter_beats(beats, filters=filters, x=x)

    try:
        beats = align_beats(beats, apd_point=apd_point, N=N)
    except Exception:
        pass

    avg = average.average_and_interpolate([b.y for b in beats], [b.t for b in beats], N)

    # plt.figure()
    # plt.plot(avg.x, avg.y)
    # plt.show()
    pacing_avg = np.interp(
        np.linspace(
            np.min(beats[0].t),
            np.max(beats[0].t),
            N,
        ),
        beats[0].t,
        beats[0].pacing,
    )

    pacing_avg[pacing_avg <= 2.5] = 0.0
    pacing_avg[pacing_avg > 2.5] = 5.0

    return Beat(y=avg.y, t=avg.x, pacing=pacing_avg, parent=beats[0].parent)


def filter_beats(
    beats: List[Beat],
    filters: Sequence[_filters.Filters],
    x: float = 1.0,
) -> List[Beat]:
    """Filter beats based of similarities of the filters

    Parameters
    ----------
    beats : List[Beat]
        List of beats
    filters : Sequence[_filters.Filters]
        List of filters
    x : float, optional
        How many standard deviations away from the mean
        the different beats should be to be
        included, by default 1.0

    Returns
    -------
    List[Beat]
        A list of filtered beats

    Raises
    ------
    _filters.InvalidFilter
        If a filter in the list of filters is not valid.

    """
    if len(filters) == 0:
        return beats
    if len(beats) == 0:
        raise ValueError("Cannot apply filter to empty list")
    feature_list: List[List[float]] = []
    bad_indices: Set[int] = set()
    # Compute the features of all beats
    for f in filters:
        if f not in _filters.Filters.__members__:
            raise _filters.InvalidFilter(f"Invalid filter {f}")
        if f.startswith("apd"):
            apds = [beat.apd(int(f[3:])) for beat in beats]

            # If any apds are negative we should remove that beat
            if any(apd < 0 for apd in apds):
                bad_indices.union(set(np.where(apd < 0 for apd in apds)[0]))

            feature_list.append(apds)
        if f == _filters.Filters.length:
            feature_list.append([len(beat) for beat in beats])
        if f == _filters.Filters.time_to_peak:
            feature_list.append([len(beat) for beat in beats])

    indices = _filters.filter_signals(feature_list, x)
    return [beats[index] for index in indices]


def apd_slope(
    beats: List[Beat],
    factor: float,
    corrected_apd: bool = False,
) -> Tuple[float, float]:
    """Compute a linear interpolation of the apd values for each beat.
    This is useful in order to see if there is a correlation between
    the APD values and the beat number. If the resulting the slope
    is relatively close to zero there is no such dependency.

    Parameters
    ----------
    beats : List[Beat]
        List of beats
    factor : float
        The apd value
    corrected_apd : bool, optional
        Whether to use corrected or regular apd, by default False

    Returns
    -------
    Tuple[float, float]
        A tuple with the (constant, slope) for the linear interpolation.
        The slope here can be interpreted as the change in APD per minute.
    """

    apd_first_points = []
    apds = []
    for beat in beats:
        beat.ensure_time_unit("ms")
        apd_first_points.append(beat.apd_point(factor=factor)[0])
        apd = beat.capd(factor=factor) if corrected_apd else beat.apd(factor)
        apds.append(apd)

    if len(apds) > 0:
        slope, const = np.polyfit(apd_first_points, apds, deg=1)
        # Covert to dAPD / min
        slope *= 1000 * 60
    else:
        slope = np.nan
        const = np.nan

    return slope, const


class Beats(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        background_correction_method: BC = BC.none,
        zero_index: Optional[int] = None,
        background_correction_kernel: int = 0,
        backend: Backend = Backend.numba,
        intervals: Optional[List[chopping.Interval]] = None,
        chopping_options: Optional[Dict[str, float]] = None,
        force_positive: bool = False,
    ) -> None:
        """Initializer for `Beats` class

        Parameters
        ----------
        y : Array
            The amplitude of the array
        t : Array
            The time stamps
        pacing : Optional[Array], optional
            An optional array of pacing amplitude values, by default None.
            If provided, it can be used to better chop the data
            into individual beats.
        background_correction_method : BC, optional
            Method to perform background correction, by default BC.none.
            Possible methods are "full", "subtract" and "none".
            See `ap_features.background.correct_background` for more info.
        zero_index : Optional[int], optional
            Index where the value should be forced to be zero, by default None.
            This will make sure that the array at the given index is zero, and
            by subtracting the value at that index from all other values in the
            array.
        backend : Backend, optional
            Backend to use for heavy computations, by default Backend.numba.
            Possible options are "numba" and "python". Note that
            most functions will be implemented in python / numpy anyway.
        intervals : Optional[List[chopping.Interval]], optional
            Optional ist of tuples containing start and ends of each beat
            to be used for chopping, by default None.
        chopping_options : Optional[Dict[str, float]], optional
            Parameters to be used for chopping trace into individual beats,
            by default None. See `ap_features.chopping.chop_data_without_pacing`
            and `ap_features.chopping.chop_data_with_pacing` for more info.
        force_positive: bool, optional
            If True, make all values positive by setting negative values to
            zero after background correction, by default False.

        Raises
        ------
        RuntimeError
            If invalid zero_index is provided
        """
        super().__init__(y, t, pacing=pacing, backend=backend)

        self.background_correction = background.correct_background(
            x=self._t,
            y=self._y,
            method=background_correction_method,
            filter_kernel_size=background_correction_kernel,
            force_positive=force_positive,
        )

        msg = (
            "Expected shape of 't' and 'y' to be the same got "
            f"{self._t.shape}(t) and {self._y.shape}(y)"
        )
        assert self._t.shape == self._y.shape, msg

        if zero_index is not None and not utils.valid_index(self._y, zero_index):
            raise RuntimeError(
                f"Invalid zero index {zero_index} for array or size {len(y)}",
            )
        self._zero_index = zero_index

        self.chopping_options = chopping.default_chopping_options()
        if chopping_options is not None:
            self.chopping_options.update(chopping_options)
        if intervals is not None:
            self.chopping_options["intervals"] = intervals

    def as_beat(self) -> "Beat":
        """Convert trace from Beats to Beat"""
        return Beat(y=self.y, t=self.t, pacing=self.pacing)

    def plot_beats(self, ylabel: str = "", align: bool = False, fname: str = ""):
        plot.plot_beats_from_beat(self, ylabel=ylabel, align=align, fname=fname)

    @property
    def chopped_data(self) -> chopping.ChoppedData:
        if not hasattr(self, "_chopped_data"):
            self._chopped_data = self.chop_data(**self.chopping_options)
        return self._chopped_data

    def apd_slope(
        self,
        factor: float,
        corrected_apd: bool = False,
    ) -> Tuple[float, float]:
        return apd_slope(self.beats, factor=factor, corrected_apd=corrected_apd)

    def chop_data(
        self,
        threshold_factor=0.5,
        min_window=50,
        max_window=5000,
        N=None,
        extend_front=None,
        extend_end=None,
        ignore_pacing=False,
        intervals=None,
    ):
        return chopping.chop_data(
            data=self.y,
            time=self.t,
            pacing=self.pacing,
            threshold_factor=threshold_factor,
            min_window=min_window,
            max_window=max_window,
            N=N,
            extend_front=extend_front,
            extend_end=extend_end,
            ignore_pacing=ignore_pacing,
            intervals=intervals,
        )

    def correct_background(
        self,
        background_correction_method: BC,
        copy: bool = True,
    ) -> "Beats":
        f = copy_function(copy)
        return Beats(
            y=f(self.y),
            t=f(self.t),
            pacing=f(self.pacing),
            background_correction_method=background_correction_method,
            zero_index=self._zero_index,
            backend=self._backend,
            chopping_options=self.chopping_options,
            intervals=self.chopping_options.get("intervals"),
        )

    def remove_points(self, t_start: float, t_end: float) -> "Beats":
        t, y = _filters.remove_points(
            x=np.copy(self.t),
            y=np.copy(self._y),
            t_start=t_start,
            t_end=t_end,
        )
        _, pacing = _filters.remove_points(
            x=np.copy(self.t),
            y=np.copy(self.pacing),
            t_start=t_start,
            t_end=t_end,
        )
        return Beats(
            y=y,
            t=t,
            pacing=pacing,
            background_correction_method=self.background_correction.method,
            zero_index=self._zero_index,
            backend=self._backend,
            chopping_options=self.chopping_options,
            intervals=self.chopping_options.get("intervals"),
        )

    def filter(self, kernel_size: int = 3, copy: bool = True) -> "Beats":
        y = _filters.filt(self._y, kernel_size=kernel_size)
        f = copy_function(copy)
        return Beats(
            y=f(y),
            t=f(self.t),
            pacing=f(self.pacing),
            background_correction_method=self.background_correction.method,
            zero_index=self._zero_index,
            backend=self._backend,
            chopping_options=self.chopping_options,
            intervals=self.chopping_options.get("intervals"),
        )

    def remove_spikes(self, spike_duration: int) -> "Beats":
        if spike_duration == 0:
            return self
        spike_points = _filters.find_spike_points(
            self.pacing,
            spike_duration=spike_duration,
        )
        t = np.delete(self.t, spike_points)
        y = np.delete(self.y, spike_points)
        pacing = np.delete(self.pacing, spike_points)

        background_correction_method = self.background_correction.method
        return Beats(
            y=y,
            t=t,
            pacing=pacing,
            background_correction_method=background_correction_method,
            zero_index=self._zero_index,
            backend=self._backend,
            chopping_options=self.chopping_options,
            intervals=self.chopping_options.get("intervals"),
        )

    def filter_beats(
        self,
        filters: Sequence[_filters.Filters],
        x: float = 1.0,
    ) -> List[Beat]:
        """Get a subset of the chopped beats based on
        similarities in different features.

        Parameters
        ----------
        filters : Sequence[_filters.Filters]
            A list of filters that should be used for filtering
        x : float, optional
            How many standard deviations away from the mean
            the different beats should be to be
            included, by default 1.0

        Returns
        -------
        List[Beat]
            A list of filtered beats

        """
        return filter_beats(self.beats, filters=filters, x=x)

    @property
    def beating_frequencies(self) -> Sequence[float]:
        """Get the frequency for each beat

        Returns
        -------
        List[float]
            List of frequencies
        """
        signals: List[Array] = [beat.y for beat in self.beats]
        times: List[Array] = [beat.t for beat in self.beats]

        if len(signals) > 1:
            return features.beating_frequency_from_peaks(signals=signals, times=times)
        else:
            return features.beating_frequency_from_apd_line(y=self.y, time=self.t)

    @property
    def beating_frequency(self) -> float:
        """The median frequency of all beats"""
        return np.median(self.beating_frequencies)

    @property
    def beat_rate(self) -> float:
        """The beat rate, i.e number of beats
        per minute, which is simply 60 divided
        by the beating frequency
        """
        return 60 * self.beating_frequency

    @property
    def beat_rates(self) -> List[float]:
        """Beat rate for all beats

        Returns
        -------
        List[float]
            List of beat rates
        """
        return [60 * bf for bf in self.beating_frequencies]

    def average_beat(
        self,
        filters: Optional[Sequence[_filters.Filters]] = None,
        N: int = 200,
        x: float = 1.0,
        apd_point: float = 50,
    ) -> Beat:
        """Compute an average beat based on
        aligning the individual beats

        Parameters
        ----------
        filters : Optional[Sequence[_filters.Filters]], optional
            A list of filters that should be used to decide
            which beats that should be included in the
            averaging, by default None
        N : int, optional
            Length of output signal, by default 200.
            Note that the output signal will be interpolated so
            that it has this length. This is done because the
            different beats might have different lengths.
        x : float, optional
            The number of standard deviations used in the
            filtering, by default 1.0

        Returns
        -------
        Beat
            An average beat.
        """
        return average_beat(beats=self.beats, N=N, apd_point=apd_point, filters=filters, x=x)

    def aligned_beats(self, N=200) -> List[Beat]:
        return align_beats(self.beats, N=N, parent=self)

    @property
    def beats(self) -> List[Beat]:
        """Chop signal into individual beats.
        You can also pass in any options that should
        be provided to the chopping algorithm.

        Returns
        -------
        List[Beat]
            A list of chopped beats
        """
        if not hasattr(self, "_beats"):
            self._beats = chopped_data_to_beats(self.chopped_data, parent=self)
        return self._beats

    @property
    def intervals(self) -> List[Beat]:
        """A ist of time intervals for each beat"""
        return self.chopped_data.intervals

    @property
    def num_beats(self) -> int:
        return len(self.beats)

    @property
    def background(self) -> np.ndarray:
        return self.background_correction.background

    @property
    def y(self) -> np.ndarray:
        y = self.background_correction.corrected
        if self._zero_index is not None:
            y = np.subtract(y, y[self._zero_index])
        return y

    @property
    def original_y(self) -> np.ndarray:
        return super().y


class BeatCollection(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        mask: Optional[Array] = None,
        parent: Optional["BeatSeriesCollection"] = None,
        backend: Backend = Backend.numba,
    ) -> None:
        super().__init__(y, t, pacing=pacing, backend=backend)
        self._parent = parent
        msg = (
            "Expected first dimension of 'y' to be the same as shape of 't' "
            f", got {self._t.size}(t) and {self._y.shape[0]}(y)"
        )
        assert self.t.size == self.y.shape[0], msg
        assert len(self.y.shape) == 2, f"Expected shape of y to be 2D, got {len(self.y.shape)}D"

    @property
    def num_traces(self):
        return self.y.shape[-1]

    @property
    def parent(self) -> Optional["BeatSeriesCollection"]:
        """If the beat comes from a BeatSeries
        object then this will return that BeatSeries

        Returns
        -------
        BeatSeries
            The parent BeatSeries
        """
        return self._parent


def chopped_data_to_beats(
    chopped_data: chopping.ChoppedData,
    parent: Optional[Beats] = None,
) -> List[Beat]:
    """Convert a ChoppedData object to a list of Beats

    Parameters
    ----------
    chopped_data : chopping.ChoppedData
        The chopped data
    parent : Optional[Beats], optional
        Parent trace, by default None

    Returns
    -------
    List[Beat]
        List of Beats
    """
    return [
        Beat(t=t, y=y, pacing=p, parent=parent, beat_number=i)
        for i, (t, y, p) in enumerate(
            zip(chopped_data.times, chopped_data.data, chopped_data.pacing),
        )
    ]


class BeatSeriesCollection(Trace):
    pass

    @property
    def num_beats(self) -> List[int]:
        raise NotImplementedError


class State(Trace):
    def __init__(
        self,
        y: Array,
        t: Optional[Array],
        pacing: Optional[Array] = None,
        backend: Backend = Backend.numba,
    ) -> None:
        super().__init__(y, t, pacing=pacing, backend=backend)

        msg = (
            "Expected first dimension of 'y' to be the same as shape of 't' "
            f", got {self._t.size}(t) and {self._y.shape[0]}(y)"
        )
        assert self.t.size == self.y.shape[0], msg
        assert len(self.y.shape) == 2, f"Expected shape of y to be D, got {len(self.y.shape)}D"

    @property
    def num_states(self):
        return self.y.shape[1]

    def __getitem__(self, k):
        return self.y[:, k]

    @property
    def cost_terms(self):
        if self.num_states != 2:
            raise NotImplementedError

        return features.cost_terms(
            v=np.ascontiguousarray(self[0]),
            ca=np.ascontiguousarray(self[1]),
            t_v=self.t,
            t_ca=self.t,
        )


class StateCollection(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        mask: Optional[Array] = None,
        parent: Optional["StateSeriesCollection"] = None,
    ) -> None:
        # TODO: Check dimensions
        super().__init__(y=y, t=t, pacing=pacing)
        self._parent = parent

        msg = (
            "Expected first dimension of 'y' to be the same as shape of 't' "
            f", got {self._t.size}(t) and {self._y.shape[0]}(y)"
        )
        assert self.t.size == self.y.shape[0], msg
        assert len(self.y.shape) == 3, f"Expected shape of y to be 3D, got {len(self.y.shape)}D"
        self.mask = mask

    @property
    def mask(self) -> Optional[Array]:
        return self._mask

    @mask.setter
    def mask(self, mask: Optional[Array]) -> None:
        if mask is not None:
            mask = utils.numpyfy(mask)
            assert mask.size == self.num_traces
        self._mask = mask

    @property
    def num_traces(self):
        return self.y.shape[-1]

    @property
    def num_states(self):
        return self.y.shape[1]

    @property
    def parent(self) -> Optional["StateSeriesCollection"]:
        """If the beat comes from a BeatSeries
        object then this will return that BeatSeries

        Returns
        -------
        BeatSeries
            The parent BeatSeries
        """
        return self._parent

    @property
    def cost_terms(self):
        return features.all_cost_terms(arr=self.y, t=self.t, mask=self.mask)


class StateSeriesCollection(Trace):
    pass

    @property
    def num_beats(self) -> List[int]:
        raise NotImplementedError
