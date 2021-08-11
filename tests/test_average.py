import numpy as np
import pytest

import ap_features as apf


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [1, 1, 1, 1, 1]),
        ([[1, 1, 1, 1, 1], [1, 1, 1]], [1, 1, 1, 1, 1]),
        ([[1, 1, 1, 1, 1], [3, 3, 3, 3, 3]], [2, 2, 2, 2, 2]),
        ([[1, 1, 1, 1, 1], [3, 3, 3]], [2, 2, 2, 1, 1]),
        ([[1, 1, 1], [3, 3], [2, 2, 5, 5]], [2, 2, 3, 5]),
    ],
)
def test_average_list(input, expected_output):
    output = apf.average_list(input)
    assert np.isclose(output, expected_output).all()
