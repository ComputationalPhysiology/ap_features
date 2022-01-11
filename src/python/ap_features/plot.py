from functools import wraps
from typing import List

import numpy as np

from . import beat as _beat


def has_matplotlib() -> bool:
    try:
        import matplotlib.pyplot  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def require_matplotlib(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        if not has_matplotlib():
            return None
        return f(*args, **kwds)

    return wrapper


def savefig(fig, fname: str) -> None:
    """Save figure if fname is not
    an empty string

    Parameters
    ----------
    fig : matplotlib.Figure
        The figure
    fname : str
        The path
    """
    import matplotlib.pyplot as plt

    if fname != "":
        fig.savefig(fname)
    else:
        plt.show()


@require_matplotlib
def plot_beat(
    beat: _beat.Trace,
    include_pacing: bool = False,
    include_background: bool = False,
    ylabel: str = "",
    fname: str = "",
):
    import matplotlib.pyplot as plt

    y = beat.y
    if include_background:
        assert isinstance(beat, _beat.Beats), "Can only plot background for Beats"
        y = beat.original_y

    fig, ax = plt.subplots()
    (line,) = ax.plot(beat.t, y)
    ax.set_xlabel("Time (ms)")
    lines = [line]
    labels = ["trace"]

    if include_pacing:
        ax2 = ax.twinx()
        (line,) = ax2.plot(beat.t, beat.pacing, color="r")
        lines.append(line)
        labels.append("pacing")
    if include_background:
        (line,) = ax.plot(beat.t, beat.background)  # type: ignore
        lines.append(line)
        labels.append("background")

    ax.set_ylabel(ylabel)
    if len(lines) > 1:
        ax.legend(lines, labels, loc="best")

    savefig(fig, fname=fname)


def plot_beats_from_beat(
    trace: _beat.Beats,
    ylabel: str = "",
    align: bool = False,
    fname: str = "",
):
    plot_beats(trace.beats, ylabel=ylabel, align=align, fname=fname)


@require_matplotlib
def plot_beats(
    beats: List[_beat.Beat],
    ylabel: str = "",
    align: bool = False,
    fname: str = "",
) -> None:

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for beat in beats:
        t = np.copy(beat.t)
        if align:
            t[:] -= t[0]

        ax.plot(t, beat.y)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)

    savefig(fig, fname=fname)


@require_matplotlib
def poincare_from_beats(
    beats: List[_beat.Beat],
    apds: List[int],
    fname: str = "",
) -> None:
    """
    Create poincare plots for given APDs

    Arguments
    ---------
    beats : List[_beat.Beat]
        List of beats
    apds : list
        List of APDs to be used, e.g [30, 50, 80]
        will plot the APS30, APD50 and APD80.
    fname : str
        Path to filename to save the figure.
        If not provided plot will be showed and not saved

    Notes
    -----
    For more info see <http://doi.org/10.1371/journal.pcbi.1003202>

    Returns
    -------
    dict:
        A dictionary with the key being the :math:`x` in APD:math:`x`
        and the value begin the points being plotted.

    """

    if len(beats) <= 1:
        # Not possible to plot poincare plot with 1 or zero elements
        return None

    import matplotlib.pyplot as plt

    apds_points = {k: [beat.apd(k) for beat in beats] for k in apds}

    fig, ax = plt.subplots()
    for k, v in apds_points.items():
        ax.plot(v[:-1], v[1:], label=f"APD{k}", marker=".")
    ax.legend(loc="best")
    ax.grid()
    ax.set_xlabel("APD(n-1)[ms]")
    ax.set_ylabel("APD(n) [ms]")

    savefig(fig, fname=fname)
    return None


def poincare(
    trace: _beat.Beats,
    apds: List[int],
    fname: str = "",
) -> None:
    """
    Create poincare plots for given APDs

    Arguments
    ---------
    trace : beat.Beats
        The trace
    apds : list
        List of APDs to be used, e.g [30, 50, 80]
        will plot the APS30, APD50 and APD80.
    fname : str
        Path to filename to save the figure.
        If not provided plot will be showed and not saved

    Notes
    -----
    For more info see <http://doi.org/10.1371/journal.pcbi.1003202>

    Returns
    -------
    dict:
        A dictionary with the key being the :math:`x` in APD:math:`x`
        and the value begin the points being plotted.

    """
    return poincare_from_beats(trace.beats, apds=apds, fname=fname)
