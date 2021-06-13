import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import ap_features


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


def main():

    a = -0.3
    b = 1.4
    tau = 20
    Iext = 0.23
    time = np.linspace(0, 999, 1000)
    res = solve_ivp(
        fitzhugh_nagumo, [0, 1000], [0, 0], args=(a, b, tau, Iext), t_eval=time,
    )

    v = res.y[0, :]
    w = res.y[1, :]

    # Find the local minima
    p, _ = find_peaks(-v)

    # Plot al beats
    fig, ax = plt.subplots()
    ax.plot(time, v, label="$v$")
    ax.plot(time, w, label="$w$")
    ax.plot(time[p], v[p], "ro")
    ax.legend()
    ax.set_xlabel("Time [ms]")

    # Extract single beat
    x = time[p[0] : p[1] + 10]
    x -= x[0]
    y = v[p[0] : p[1] + 10]

    # Plot single beat
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid()

    # Get cost terms
    cost_terms = ap_features.cost_terms_trace_c(y, x)
    labels = ap_features.list_cost_function_terms_trace()

    for c, l in zip(cost_terms, labels):
        print(f"{l:20}:{c:10.3f}")

    # Plot cost terms
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(cost_terms))
    ax.bar(x, cost_terms)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
