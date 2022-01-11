from matplotlib.testing.decorators import image_comparison

from ap_features import plot


@image_comparison(baseline_images=["poincare"], extensions=["png"])
def test_poincare_from_beats_plot(real_beats):
    beats = real_beats.beats
    plot.poincare_from_beats(beats, [30, 50, 70, 80])


@image_comparison(baseline_images=["poincare"], extensions=["png"])
def test_poincare_plot(real_beats):
    plot.poincare(real_beats, [30, 50, 70, 80])


@image_comparison(baseline_images=["simple_beat"], extensions=["png"])
def test_plot_beat_no_pacing(real_beats):
    plot.plot_beat(real_beats)


@image_comparison(baseline_images=["simple_beat_with_pacing"], extensions=["png"])
def test_plot_beat_with_pacing(real_beats):
    plot.plot_beat(real_beats, include_pacing=True)
