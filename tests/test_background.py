import itertools as it

import numpy as np
import pytest
from ap_features import background


# Test linear and quadratic background for each method
@pytest.mark.parametrize(
    "a, method",
    it.product([0, 0.1], background.BackgroundCorrection),
)
def test_background(a, method):
    N = 700
    x = np.linspace(0, 7, N)
    y = np.sin(2 * np.pi * 1.2 * x) + 1

    b = -1.5
    c = 100.0

    bkg = a * x ** 2 + b * x + c
    signal = y + bkg

    estimated_background = background.background(x, signal)
    corrected = background.correct_background(x, signal, method=method)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(x, signal, label="signal")
    # ax[0].plot(x, bkg, linestyle="--", label="true")

    # ax[0].plot(x, estimated_background, label="estimated")
    # ax[1].plot(x, corrected.corrected, label="corrected")
    # ax[1].plot(x, y, linestyle="--", label="true")
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()

    assert np.isclose(estimated_background, bkg, rtol=1e-3).all()
    if method == background.BackgroundCorrection.none:
        assert np.isclose(corrected.corrected, signal).all()
    elif method == background.BackgroundCorrection.subtract:
        assert np.isclose(corrected.corrected, y, atol=1e-1).all()
    else:
        assert np.isclose(corrected.corrected, y / bkg[0], atol=1e-1).all()


def test_invalid_backgroud_raises_ValueError():
    with pytest.raises(ValueError):
        background.correct_background(np.zeros(10), np.zeros(10), method="gmsadokg")


def test_different_length_raises_ValueError():
    with pytest.raises(ValueError):
        background.correct_background(np.zeros(10), np.zeros(9), method="full")


if __name__ == "__main__":
    test_background(0.0, "subtract")
