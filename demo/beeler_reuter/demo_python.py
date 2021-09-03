import ap_features as apf
import beeler_reuter_1977_version06 as model
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def main():

    states = model.init_state_values()
    parameters = model.init_parameter_values()

    num_beats = 1
    T = num_beats * 1000
    dt = 1.0
    time = np.arange(0, T, dt)

    res = solve_ivp(
        model.rhs,
        [0, T],
        states,
        args=(parameters,),
        t_eval=time,
    )

    V = res.y[model.state_indices("V"), :]
    Cai = res.y[model.state_indices("Cai"), :]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, V)
    ax[1].plot(time, Cai)
    ax[0].set_ylabel("Voltage")
    ax[1].set_ylabel("Intracellular calcium")
    ax[1].set_xlabel("Time [ms]")
    ax[0].set_title("Beeler Reuter model")

    beat = apf.Beat(y=Cai, t=time)
    # beat = apf.Beat(y=V, t=time)

    # Plot cost terms
    fig, ax = plt.subplots(figsize=(12, 6))
    cost_terms = apf.list_cost_function_terms_trace()
    x = np.arange(len(cost_terms))
    ax.bar(x=x, height=beat.cost_terms, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cost_terms, rotation=30)
    ax.grid()
    ax.set_title("Action potential duration for different beats")

    plt.show()


if __name__ == "__main__":
    main()
