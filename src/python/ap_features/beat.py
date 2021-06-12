from typing import List, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import background, features
from .utils import Array


class Trace:
    def __init__(
        self,
        t: Array,
        y: Array,
        pacing: Optional[Array] = None,
    ) -> None:

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
        return f"{self.__class__.__name__}(t={self.t}, y={self.y})"

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
        t: Array,
        y: Array,
        pacing: Optional[Array] = None,
        y_rest: Optional[float] = None,
        parent: Optional["BeatSeries"] = None,
    ) -> None:

        super().__init__(t, y, pacing)
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

    def apd_up(self, factor_x, factor_y):
        pass


class BeatSeries(Trace):
    def __init__(
        self,
        t: Array,
        y: Array,
        pacing: Optional[Array] = None,
        correct_background: bool = False,
    ) -> None:
        self._background = None
        if correct_background:
            self._background = background.correct_background(x=t, y=y)

        super().__init__(t, y, pacing=pacing)

    def chop(self) -> List[Beat]:
        raise NotImplementedError

    def beatrate(self):
        raise NotImplementedError

    @property
    def background(self) -> Optional[background.Background]:
        return self._background
