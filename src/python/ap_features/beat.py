from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import background
from . import chopping
from . import features
from .utils import Array
from .utils import Backend
from .utils import normalize_signal
from .utils import numpyfy


class Trace:
    def __init__(
        self,
        y: Array,
        t: Optional[Array],
        pacing: Optional[Array] = None,
        backend: Backend = Backend.c,
    ) -> None:

        if t is None:
            t = np.arange(len(y))
        self._t = numpyfy(t)
        self._y = numpyfy(y)
        if pacing is None:
            pacing = np.zeros_like(self._t)  # type: ignore

        self._pacing = numpyfy(pacing)

        assert backend in Backend
        self._backend = backend

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(t={self.t.shape}, y={self.y.shape})"

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def pacing(self) -> np.ndarray:
        return self._pacing


class Beat(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        y_rest: Optional[float] = None,
        parent: Optional["Beats"] = None,
        backend: Backend = Backend.c,
    ) -> None:

        super().__init__(y, t, pacing=pacing, backend=backend)
        msg = (
            "Expected shape of 't' and 'y' to be the same. got "
            f"{self._t.shape}(t) and {self._y.shape}(y)"
        )
        assert self._t.shape == self._y.shape, msg
        self._y_rest = y_rest
        self._parent = parent

    @property
    def y_normalized(self):
        return normalize_signal(self.y, self.y_rest)

    @property
    def y_rest(self):
        return self._y_rest

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

    def apd(self, factor: int) -> Optional[float]:
        """The action potential duration

        Parameters
        ----------
        factor : int
            Integer between 0 and 100

        Returns
        -------
        float
            action potential duration
        """
        return features.apd(factor=factor, V=self.y, t=self.t, v_r=self.y_rest)

    def tau(self, a: float) -> float:
        """Decay time. Time for the signal amplitude to go from maxium to
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

    def apd_up(self, factor_x, factor_y):
        pass

    @property
    def cost_terms(self):
        return features.cost_terms_trace(y=self.y, t=self.t, backend=self._backend)


class Beats(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        correct_background: bool = False,
        backend: Backend = Backend.c,
    ) -> None:
        self._background = None
        if correct_background:
            self._background = background.correct_background(x=t, y=y)

        super().__init__(y, t, pacing=pacing, backend=backend)
        msg = (
            "Expected shape of 't' and 'y' to be the same got "
            f"{self._t.shape}(t) and {self._y.shape}(y)"
        )
        assert self._t.shape == self._y.shape, msg

    def chop(self, **options) -> List[Beat]:
        c = chopping.chop_data(data=self.y, time=self.t, pacing=self.pacing, **options)
        self._beats = [
            Beat(t=t, y=y, pacing=p) for (t, y, p) in zip(c.times, c.data, c.pacing)
        ]
        self._chopped_data = c
        return self._beats

    @property
    def beating_frequencies(self) -> Sequence[float]:
        signals: List[Array] = [beat.y for beat in self.beats]
        times: List[Array] = [beat.t for beat in self.beats]
        return features.beating_frequency_from_peaks(signals=signals, times=times)

    @property
    def beating_frequency(self) -> float:
        return np.median(self.beating_frequencies)

    @property
    def beat_rate(self) -> float:
        return 60 / self.beating_frequency

    @property
    def beat_rates(self) -> List[float]:
        return [60 / bf for bf in self.beating_frequencies]

    @property
    def beats(self) -> List[Beat]:
        if not hasattr(self, "_beats"):
            self.chop()
        return self._beats

    @property
    def num_beats(self) -> int:
        return len(self.beats)

    @property
    def background(self) -> Optional[background.Background]:
        return self._background

    @property
    def y(self) -> np.ndarray:
        if self.background is not None:
            return self.background.corrected
        return super().y

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
        backend: Backend = Backend.c,
    ) -> None:

        super().__init__(y, t, pacing=pacing, backend=backend)
        self._parent = parent
        msg = (
            "Expected first dimension of 'y' to be the same as shape of 't' "
            f", got {self._t.size}(t) and {self._y.shape[0]}(y)"
        )
        assert self.t.size == self.y.shape[0], msg
        assert (
            len(self.y.shape) == 2
        ), f"Expected shape of y to be 2D, got {len(self.y.shape)}D"

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
        backend: Backend = Backend.c,
    ) -> None:
        super().__init__(y, t, pacing=pacing, backend=backend)

        msg = (
            "Expected first dimension of 'y' to be the same as shape of 't' "
            f", got {self._t.size}(t) and {self._y.shape[0]}(y)"
        )
        assert self.t.size == self.y.shape[0], msg
        assert (
            len(self.y.shape) == 2
        ), f"Expected shape of y to be D, got {len(self.y.shape)}D"

    @property
    def num_states(self):
        return self.y.shape[1]

    def __getitem__(self, k):
        return self.y[:, k]

    @property
    def cost_terms(self):
        if self.num_states != 2:
            raise NotImplementedError

        return features.cost_terms(v=self[0], ca=self[1], t_v=self.t, t_ca=self.t)


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
        assert (
            len(self.y.shape) == 3
        ), f"Expected shape of y to be 3D, got {len(self.y.shape)}D"
        self.mask = mask

    @property
    def mask(self) -> Optional[Array]:
        return self._mask

    @mask.setter
    def mask(self, mask: Optional[Array]) -> None:
        if mask is not None:
            mask = numpyfy(mask)
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
