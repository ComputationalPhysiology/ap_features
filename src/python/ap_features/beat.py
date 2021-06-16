from typing import List, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import background, chopping, features
from .utils import Array


class Trace:
    def __init__(
        self,
        y: Array,
        t: Optional[Array],
        pacing: Optional[Array] = None,
    ) -> None:

        if t is None:
            t = np.arange(len(y))
        self._t = np.array(t)
        self._y = np.array(y)
        if pacing is None:
            pacing = np.zeros_like(self._y)  # type: ignore

        msg = (
            "Expected shape of 't' and 'y' to be the same got "
            f"{self._t.shape}(t) and {self._y.shape}(y)"
        )
        assert self._t.shape == self._y.shape, msg
        self._pacing = np.array(pacing)

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
        parent: Optional["BeatSeries"] = None,
    ) -> None:

        super().__init__(y, t, pacing)
        self._y_rest = y_rest
        self._parent = parent

    @property
    def y_normalized(self):
        return features.normalize_signal(self.y, self.y_rest)

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
    def parent(self) -> "BeatSeries":
        """If the beat comes from a BeatSeries
        object then this will return that BeatSeries

        Returns
        -------
        BeatSeries
            The parent BeatSeries
        """

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


class BeatSeries(Trace):
    def __init__(
        self,
        y: Array,
        t: Array,
        pacing: Optional[Array] = None,
        correct_background: bool = False,
    ) -> None:
        self._background = None
        if correct_background:
            self._background = background.correct_background(x=t, y=y)

        super().__init__(y, t, pacing=pacing)

    def chop(self, **options) -> List[Beat]:
        c = chopping.chop_data(data=self.y, time=self.t, pacing=self.pacing, **options)
        self._beats = [
            Beat(t=t, y=y, pacing=p) for (t, y, p) in zip(c.times, c.data, c.pacing)
        ]
        self._chopped_data = c
        return self._beats

    def beatrate(self):
        raise NotImplementedError

    @property
    def beats(self) -> List[Beat]:
        if not hasattr(self, "_beats"):
            raise ValueError("Please chop BeatSeries into Beats first")
        return self._beats

    @property
    def num_beats(self) -> int:
        return len(self._beats)

    @property
    def background(self) -> Optional[background.Background]:
        return self._background
