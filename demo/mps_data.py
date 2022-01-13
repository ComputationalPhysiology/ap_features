import ap_features as apf
import matplotlib.pyplot as plt
import mps


def print_beats_info(beats):
    print(beats.num_beats)
    print(beats.beating_frequency)
    print(beats.beat_rate)


def plot_beats(beats):
    fig, ax = plt.subplots(3, 2, figsize=(8, 10))
    ax[0, 0].plot(beats.t, beats.original_y)
    ax2 = ax[0, 0].twinx()
    ax2.plot(beats.t, beats.pacing, color="r")
    ax[0, 0].set_title("Original data")

    ax[0, 1].plot(beats.t, beats.original_y)
    if beats.background is not None:
        ax[0, 1].plot(beats.t, beats.background.background)
    ax[0, 1].set_title("Original data with background")

    ax[1, 0].plot(beats.t, beats.y)
    ax[1, 0].set_title("Corrected data")

    for i, beat in enumerate(beats.beats, start=1):
        ax[1, 1].plot(beat.t, beat.y, label=f"Beat {i}")
    ax[1, 1].legend()
    ax[1, 1].set_title("Individual beats")

    for i, beat in enumerate(beats.beats, start=1):
        ax[2, 0].plot(beat.t - beat.t[0], beat.y, label=f"Beat {i}")
    ax[2, 0].legend()
    ax[2, 0].set_title("Chopped beats (aligned)")

    avg = beats.average_beat()
    ax[2, 1].plot(avg.t, avg.y)
    ax[2, 1].set_title("Average")

    fig.savefig("mps_data_paced_corrected.png")


def main():

    path = "/home/henriknf/data/mps/181116_Lidocaine/20181116_0uM_1hz/Point1A_MM_ChannelCyan_VC_Seq0001.nd2"
    # path = "/home/henriknf/data/mps/181116_Lidocaine/20181116_0uM_spont/Point1A_MM_ChannelCyan_VC_Seq0002.nd2"
    data = mps.MPS(path)

    y = data.frames.mean((0, 1))
    t = data.time_stamps
    pacing = data.pacing

    beats = apf.Beats(y, t, pacing, correct_background=True)
    print_beats_info(beats)
    plot_beats(beats)


if __name__ == "__main__":
    main()
