# Gotran generated code for the "beeler_reuter_1977_version06" model
from __future__ import division

import math

import numpy as np


def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    # m=0.011, h=0.988, j=0.975, Cai=0.0001, d=0.003, f=0.994, x1=0.0001,
    # V=-84.624
    init_values = np.array(
        [0.011, 0.988, 0.975, 0.0001, 0.003, 0.994, 0.0001, -84.624],
        dtype=np.float_,
    )

    # State indices and limit checker
    state_ind = dict(
        [
            ("m", 0),
            ("h", 1),
            ("j", 2),
            ("Cai", 3),
            ("d", 4),
            ("f", 5),
            ("x1", 6),
            ("V", 7),
        ],
    )

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind = state_ind[state_name]

        # Assign value
        init_values[ind] = value

    return init_values


def init_parameter_values(**values):
    """
    Initialize parameter values
    """
    # Param values
    # E_Na=50.0, g_Na=0.04, g_Nac=3e-05, g_s=0.0009, IstimAmplitude=0.5,
    # IstimEnd=50000.0, IstimPeriod=1000.0, IstimPulseDuration=1.0,
    # IstimStart=10.0, C=0.01
    init_values = np.array(
        [50.0, 0.04, 3e-05, 0.0009, 0.5, 50000.0, 1000.0, 1.0, 10.0, 0.01],
        dtype=np.float_,
    )

    # Parameter indices and limit checker
    param_ind = dict(
        [
            ("E_Na", 0),
            ("g_Na", 1),
            ("g_Nac", 2),
            ("g_s", 3),
            ("IstimAmplitude", 4),
            ("IstimEnd", 5),
            ("IstimPeriod", 6),
            ("IstimPulseDuration", 7),
            ("IstimStart", 8),
            ("C", 9),
        ],
    )

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind = param_ind[param_name]

        # Assign value
        init_values[ind] = value

    return init_values


def state_indices(*states):
    """
    State indices
    """
    state_inds = dict(
        [
            ("m", 0),
            ("h", 1),
            ("j", 2),
            ("Cai", 3),
            ("d", 4),
            ("f", 5),
            ("x1", 6),
            ("V", 7),
        ],
    )

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices) > 1:
        return indices
    else:
        return indices[0]


def parameter_indices(*params):
    """
    Parameter indices
    """
    param_inds = dict(
        [
            ("E_Na", 0),
            ("g_Na", 1),
            ("g_Nac", 2),
            ("g_s", 3),
            ("IstimAmplitude", 4),
            ("IstimEnd", 5),
            ("IstimPeriod", 6),
            ("IstimPulseDuration", 7),
            ("IstimStart", 8),
            ("C", 9),
        ],
    )

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices) > 1:
        return indices
    else:
        return indices[0]


def monitor_indices(*monitored):
    """
    Monitor indices
    """
    monitor_inds = dict(
        [
            ("i_Na", 0),
            ("alpha_m", 1),
            ("beta_m", 2),
            ("alpha_h", 3),
            ("beta_h", 4),
            ("alpha_j", 5),
            ("beta_j", 6),
            ("E_s", 7),
            ("i_s", 8),
            ("alpha_d", 9),
            ("beta_d", 10),
            ("alpha_f", 11),
            ("beta_f", 12),
            ("i_x1", 13),
            ("alpha_x1", 14),
            ("beta_x1", 15),
            ("Istim", 16),
            ("i_K1", 17),
            ("dm_dt", 18),
            ("dh_dt", 19),
            ("dj_dt", 20),
            ("dCai_dt", 21),
            ("dd_dt", 22),
            ("df_dt", 23),
            ("dx1_dt", 24),
            ("dV_dt", 25),
        ],
    )

    indices = []
    for monitor in monitored:
        if monitor not in monitor_inds:
            raise ValueError("Unknown monitored: '{0}'".format(monitor))
        indices.append(monitor_inds[monitor])
    if len(indices) > 1:
        return indices
    else:
        return indices[0]


def rhs(t, states, parameters, values=None):
    """
    Compute the right hand side of the beeler_reuter_1977_version06 ODE
    """

    # Assign states
    assert len(states) == 8
    m, h, j, Cai, d, f, x1, V = states

    # Assign parameters
    assert len(parameters) == 10
    (
        E_Na,
        g_Na,
        g_Nac,
        g_s,
        IstimAmplitude,
        IstimEnd,
        IstimPeriod,
        IstimPulseDuration,
        IstimStart,
        C,
    ) = parameters

    # Init return args
    if values is None:
        values = np.zeros((8,), dtype=np.float_)
    else:
        assert isinstance(values, np.ndarray) and values.shape == (8,)

    # Expressions for the Sodium current component
    i_Na = (g_Nac + g_Na * (m * m * m) * h * j) * (-E_Na + V)

    # Expressions for the m gate component
    alpha_m = (-47 - V) / (-1 + 0.009095277101695816 * math.exp(-0.1 * V))
    beta_m = 0.7095526727489909 * math.exp(-0.056 * V)
    values[0] = (1 - m) * alpha_m - beta_m * m

    # Expressions for the h gate component
    alpha_h = 5.497962438709065e-10 * math.exp(-0.25 * V)
    beta_h = 1.7 / (1 + 0.1580253208896478 * math.exp(-0.082 * V))
    values[1] = (1 - h) * alpha_h - beta_h * h

    # Expressions for the j gate component
    alpha_j = (
        1.8690473007222892e-10
        * math.exp(-0.25 * V)
        / (1 + 1.6788275299956603e-07 * math.exp(-0.2 * V))
    )
    beta_j = 0.3 / (1 + 0.040762203978366204 * math.exp(-0.1 * V))
    values[2] = (1 - j) * alpha_j - beta_j * j

    # Expressions for the Slow inward current component
    E_s = -82.3 - 13.0287 * math.log(0.001 * Cai)
    i_s = g_s * (-E_s + V) * d * f
    values[3] = 7.000000000000001e-06 - 0.07 * Cai - 0.01 * i_s

    # Expressions for the d gate component
    alpha_d = (
        0.095
        * math.exp(1 / 20 - V / 100)
        / (1 + 1.4332881385696572 * math.exp(-0.07199424046076314 * V))
    )
    beta_d = 0.07 * math.exp(-44 / 59 - V / 59) / (1 + math.exp(11 / 5 + V / 20))
    values[4] = (1 - d) * alpha_d - beta_d * d

    # Expressions for the f gate component
    alpha_f = (
        0.012
        * math.exp(-28 / 125 - V / 125)
        / (1 + 66.5465065250986 * math.exp(0.14992503748125938 * V))
    )
    beta_f = 0.0065 * math.exp(-3 / 5 - V / 50) / (1 + math.exp(-6 - V / 5))
    values[5] = (1 - f) * alpha_f - beta_f * f

    # Expressions for the Time dependent outward current component
    i_x1 = (
        0.0019727757115328517
        * (-1 + 21.75840239619708 * math.exp(0.04 * V))
        * math.exp(-0.04 * V)
        * x1
    )

    # Expressions for the X1 gate component
    alpha_x1 = (
        0.031158410986342627
        * math.exp(0.08264462809917356 * V)
        / (1 + 17.41170806332765 * math.exp(0.05714285714285714 * V))
    )
    beta_x1 = (
        0.0003916464405623223
        * math.exp(-0.05998800239952009 * V)
        / (1 + math.exp(-4 / 5 - V / 25))
    )
    values[6] = (1 - x1) * alpha_x1 - beta_x1 * x1

    # Expressions for the Time independent outward current component
    i_K1 = 0.0035 * (4.6000000000000005 + 0.2 * V) / (
        1 - 0.39851904108451414 * math.exp(-0.04 * V)
    ) + 0.0035 * (-4 + 119.85640018958804 * math.exp(0.04 * V)) / (
        8.331137487687693 * math.exp(0.04 * V) + 69.4078518387552 * math.exp(0.08 * V)
    )

    # Expressions for the Stimulus protocol component
    Istim = (
        IstimAmplitude
        if t - IstimStart - IstimPeriod * math.floor((t - IstimStart) / IstimPeriod)
        <= IstimPulseDuration
        and t <= IstimEnd
        and t >= IstimStart
        else 0
    )

    # Expressions for the Membrane component
    values[7] = (-i_K1 - i_Na - i_s - i_x1 + Istim) / C

    # Return results
    return values


def monitor(states, t, parameters, monitored=None):
    """
    Computes monitored expressions of the beeler_reuter_1977_version06 ODE
    """

    # Assign states
    assert len(states) == 8
    m, h, j, Cai, d, f, x1, V = states

    # Assign parameters
    assert len(parameters) == 10
    (
        E_Na,
        g_Na,
        g_Nac,
        g_s,
        IstimAmplitude,
        IstimEnd,
        IstimPeriod,
        IstimPulseDuration,
        IstimStart,
        C,
    ) = parameters

    # Init return args
    if monitored is None:
        monitored = np.zeros((26,), dtype=np.float_)
    else:
        assert isinstance(monitored, np.ndarray) and monitored.shape == (26,)

    # Expressions for the Sodium current component
    monitored[0] = (g_Nac + g_Na * (m * m * m) * h * j) * (-E_Na + V)

    # Expressions for the m gate component
    monitored[1] = (-47 - V) / (-1 + 0.009095277101695816 * math.exp(-0.1 * V))
    monitored[2] = 0.7095526727489909 * math.exp(-0.056 * V)
    monitored[18] = (1 - m) * monitored[1] - m * monitored[2]

    # Expressions for the h gate component
    monitored[3] = 5.497962438709065e-10 * math.exp(-0.25 * V)
    monitored[4] = 1.7 / (1 + 0.1580253208896478 * math.exp(-0.082 * V))
    monitored[19] = (1 - h) * monitored[3] - h * monitored[4]

    # Expressions for the j gate component
    monitored[5] = (
        1.8690473007222892e-10
        * math.exp(-0.25 * V)
        / (1 + 1.6788275299956603e-07 * math.exp(-0.2 * V))
    )
    monitored[6] = 0.3 / (1 + 0.040762203978366204 * math.exp(-0.1 * V))
    monitored[20] = (1 - j) * monitored[5] - j * monitored[6]

    # Expressions for the Slow inward current component
    monitored[7] = -82.3 - 13.0287 * math.log(0.001 * Cai)
    monitored[8] = g_s * (-monitored[7] + V) * d * f
    monitored[21] = 7.000000000000001e-06 - 0.07 * Cai - 0.01 * monitored[8]

    # Expressions for the d gate component
    monitored[9] = (
        0.095
        * math.exp(1 / 20 - V / 100)
        / (1 + 1.4332881385696572 * math.exp(-0.07199424046076314 * V))
    )
    monitored[10] = 0.07 * math.exp(-44 / 59 - V / 59) / (1 + math.exp(11 / 5 + V / 20))
    monitored[22] = (1 - d) * monitored[9] - d * monitored[10]

    # Expressions for the f gate component
    monitored[11] = (
        0.012
        * math.exp(-28 / 125 - V / 125)
        / (1 + 66.5465065250986 * math.exp(0.14992503748125938 * V))
    )
    monitored[12] = 0.0065 * math.exp(-3 / 5 - V / 50) / (1 + math.exp(-6 - V / 5))
    monitored[23] = (1 - f) * monitored[11] - f * monitored[12]

    # Expressions for the Time dependent outward current component
    monitored[13] = (
        0.0019727757115328517
        * (-1 + 21.75840239619708 * math.exp(0.04 * V))
        * math.exp(-0.04 * V)
        * x1
    )

    # Expressions for the X1 gate component
    monitored[14] = (
        0.031158410986342627
        * math.exp(0.08264462809917356 * V)
        / (1 + 17.41170806332765 * math.exp(0.05714285714285714 * V))
    )
    monitored[15] = (
        0.0003916464405623223
        * math.exp(-0.05998800239952009 * V)
        / (1 + math.exp(-4 / 5 - V / 25))
    )
    monitored[24] = (1 - x1) * monitored[14] - monitored[15] * x1

    # Expressions for the Time independent outward current component
    monitored[17] = 0.0035 * (4.6000000000000005 + 0.2 * V) / (
        1 - 0.39851904108451414 * math.exp(-0.04 * V)
    ) + 0.0035 * (-4 + 119.85640018958804 * math.exp(0.04 * V)) / (
        8.331137487687693 * math.exp(0.04 * V) + 69.4078518387552 * math.exp(0.08 * V)
    )

    # Expressions for the Stimulus protocol component
    monitored[16] = (
        IstimAmplitude
        if t - IstimStart - IstimPeriod * math.floor((t - IstimStart) / IstimPeriod)
        <= IstimPulseDuration
        and t <= IstimEnd
        and t >= IstimStart
        else 0
    )

    # Expressions for the Membrane component
    monitored[25] = (
        -monitored[0] - monitored[13] - monitored[17] - monitored[8] + monitored[16]
    ) / C

    # Return results
    return monitored
