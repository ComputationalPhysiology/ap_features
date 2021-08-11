import ap_features as apf
import numpy as np
import pytest


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


def test_empty_average():
    avg = apf.average_and_interpolate([], [])
    assert avg == ([], [], [], [])


@pytest.mark.parametrize(
    "length, num_beats",
    [(1, 1), (2, 2), (1, 0), (0, 1), (5, 10), (10, 5)],
)
def test_average_and_interpolate_same_length(length, num_beats):

    xs = [np.arange(length) for i in range(num_beats)]
    ys = [i * np.ones(length) for i in range(num_beats)]
    avg = apf.average_and_interpolate(ys, xs, N=length)

    x = [] if len(xs) == 0 else xs[0]
    assert np.isclose(avg.x, x).all()
    assert np.isclose(avg.y, np.mean(ys, 0)).all()
    assert len(avg.y) == len(avg.x)


@pytest.mark.parametrize("length, num_beats", [(1, 1), (2, 2), (5, 10), (10, 5)])
def test_average_and_interpolate_different_length(length, num_beats):
    N = 200
    xs = [np.arange(length) for i in range(num_beats)]
    ys = [i * np.ones(length) for i in range(num_beats)]
    avg = apf.average_and_interpolate(ys, xs, N=N)

    assert len(avg.y) == len(avg.x) == N
    # Check monotonicity
    assert (np.diff(avg.x) >= 0).all()
    # Check boundaries of x
    assert np.isclose(avg.x[0], np.min(xs)).all()
    assert np.isclose(avg.x[-1], np.max(xs)).all()
    # Check y
    assert np.isclose(avg.y, np.mean(ys)).all()


def test_raises_Error_on_differet_number_of_signals():
    with pytest.raises(apf.average.InvalidSubSignalError):
        apf.average_and_interpolate([[1], [1]], [[1]])


def test_raises_Error_on_differen_signal_lengths():
    with pytest.raises(apf.average.InvalidSubSignalError):
        apf.average_and_interpolate([[1], [1, 2]], [[1], [2]])
