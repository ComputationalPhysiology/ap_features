from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

here = Path(__file__).absolute().parent


def fitzhugh_nagumo(t, x, a, b, tau, Iext):
    """Time derivative of the Fitzhugh-Nagumo neural model.
    Parameters

    Parameters
    ----------
    t : float
        Time (not used)
    x : np.ndarray
        State of size 2 - (Membrane potential, Recovery variable)
    a : float
        Parameter in the model
    b : float
        Parameter in the model
    tau : float
        Time scale
    Iext : float
        Constant stimulus current

    Returns
    -------
    np.ndarray
        dx/dt - size 2
    """
    return np.array([x[0] - x[0] ** 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])


@pytest.fixture(scope="session")
def multiple_beats():

    a = -0.3
    b = 1.4
    tau = 20
    Iext = 0.23
    time = np.linspace(0, 999, 1000)
    res = solve_ivp(
        fitzhugh_nagumo,
        [0, 1000],
        [0, 0],
        args=(a, b, tau, Iext),
        t_eval=time,
    )
    return time, res.y[0, :]


@pytest.fixture(scope="session")
def single_beat(multiple_beats):
    time, v = multiple_beats
    # Find the local minima
    p, _ = find_peaks(-v)
    x = time[p[0] : p[1] + 10]
    x -= x[0]
    y = v[p[0] : p[1] + 10]
    return x, y


@pytest.fixture(scope="session")
def triangle_signal():
    x = np.arange(301, dtype=float)

    y = np.zeros(301, dtype=float)
    y[:101] = np.linspace(0, 1, 101)
    y[101:201] = np.linspace(1, 0, 101)[1:]
    return (x, y)


@pytest.fixture(scope="session")
def synthetic_data(num_parameter_sets=1):
    data = np.load(here.joinpath("data.npy"), allow_pickle=True).item()
    cost = np.load(here.joinpath("cost_terms.npy"))
    trace = np.array([data["v"], data["ca"]])
    return trace, data["t"], cost
