from typing import List, Optional, Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import features


class Trace:
    def __init__(
        self,
        t: Sequence[float],
        y: Sequence[float],
        pacing: Optional[Sequence[float]] = None,
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
        t: Sequence[float],
        y: Sequence[float],
        pacing: Optional[Sequence[float]] = None,
        y_rest: float = 0,
    ) -> None:

        super().__init__(t, y, pacing)
        self._y_rest = y_rest

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

    def apd(self, factor):
        return features.apd(factor=factor, V=self.y, t=self.t, v_r=self.y_rest)

    def apd_up(self, factor_x, factor_y):
        pass


class BeatSeries(Trace):
    def chop(self) -> List[Beat]:
        raise NotImplementedError

    def beatrate(self):
        raise NotImplementedError
