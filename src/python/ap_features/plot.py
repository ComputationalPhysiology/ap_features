from functools import wraps
from typing import List
from typing import Union

from . import beat


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


@require_matplotlib
def plot_beat(
    beat: Union[beat.Beat, beat.Beats],
    include_pacing: bool = False,
    fname: str = "",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(beat.t, beat.y)
    ax.set_xlabel("Time (ms)")
    if include_pacing:
        ax2 = ax.twinx()
        ax2.plot(beat.t, beat.pacing, color="r")
    fig.savefig(fname)
    if fname != "":
        fig.savefig(fname)
    else:
        plt.show()


@require_matplotlib
def poincare_from_beats(
    beats: List[beat.Beat],
    apds: List[int],
    fname: str = "",
) -> None:
    """
    Create poincare plots for given APDs

    Arguments
    ---------
    beats : List[beat.Beat]
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
    if fname != "":
        fig.savefig(fname)
    else:
        plt.show()
    return None


def poincare(
    trace: beat.Beats,
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
