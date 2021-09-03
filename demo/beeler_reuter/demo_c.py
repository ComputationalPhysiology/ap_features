import ap_features as apf
import cmodel
import matplotlib.pyplot as plt
import numpy as np


def main():

    lib = np.ctypeslib.load_library("libbeeler_reuter.dylib", "build/lib")
    model = cmodel.CModel(lib, "beeler_reuter_1977_version06.ode")

    num_beats = 10
    T = num_beats * 1000
    dt = 1.0

    time, y = model.solve(
        t_start=0,
        t_end=T,
        dt=dt,
    )
    V = y[:, model.state_index("V")]
    Cai = y[:, model.state_index("Cai")]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, V)
    ax[1].plot(time, Cai)
    ax[0].set_ylabel("Voltage")
    ax[1].set_ylabel("Intracellular calcium")
    ax[1].set_xlabel("Time [ms]")
    ax[0].set_title("Beeler Reuter model")

    s = apf.Beats(y=V, t=time)
    beats = s.chop()
    fig, ax = plt.subplots()
    for beat in beats:
        ax.plot(beat.t, beat.y)
    ax.set_title("Chopped beats")

    # Plot some APDs
    fig, ax = plt.subplots()
    apds = [20, 40, 50, 70, 80]
    N = len(apds)
    x = np.arange(N)
    width = 1 / (s.num_beats + 1)
    for i, beat in enumerate(beats):
        ax.bar(x + i * width, [beat.apd(apd) for apd in apds], width=width)
    ax.set_xticks(x + 0.5 - width)
    ax.set_xticklabels(apds)
    ax.set_ylabel("Time [ms]")
    ax.set_xlabel("APD")
    ax.grid()
    ax.set_title("Action potential duration for different beats")

    plt.show()


if __name__ == "__main__":
    main()
