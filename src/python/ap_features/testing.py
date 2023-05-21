import numpy as np


def ca_transient(t, tstart=0.05):
    tau1 = 0.05
    tau2 = 0.110

    ca_diast = 0.0
    ca_ampl = 1.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1) - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca


def fitzhugh_nagumo(t, x, a=-0.3, b=1.4, tau=20.0, Iext=0.23):
    """Time derivative of the Fitzhugh-Nagumo neural model.
    Parameters

    Parameters
    ----------
    t : float
        Time (not used)
    x : np.ndarray
        State of size 2 - (Membrane potential, Recovery variable)
    a : float
        Parameter in the model, by default -0.3
    b : float
        Parameter in the model, by default 1.4
    tau : float
        Time scale, by default 20.0
    Iext : float
        Constant stimulus current, by default 0.23

    Returns
    -------
    np.ndarray
        dx/dt - size 2
    """
    return np.array([x[0] - x[0] ** 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])
