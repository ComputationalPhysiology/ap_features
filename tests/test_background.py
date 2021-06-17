import numpy as np
import pytest
from ap_features import background


# Test linear and quadratic background
@pytest.mark.parametrize("a", [0, 0.1])
def test_background(a):
    N = 700
    x = np.linspace(0, 7, N)
    y = np.sin(2 * np.pi * 1.2 * x)

    b = -1.5
    c = 10.0

    bkg = a * x ** 2 + b * x + c
    bkg -= bkg[0]
    signal = y + bkg
    estimated_background = background.background(x, signal)
    corrected = background.correct_background(x, signal)

    z1 = bkg - bkg[-1]
    z2 = estimated_background - estimated_background[-1]

    print(abs(np.subtract(z1, z2)))
    assert all(abs(np.subtract(z1, z2)) < 1e-2)
    assert all(abs(np.subtract(corrected.corrected, y)))
