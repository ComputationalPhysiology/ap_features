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
def test_average_list_without_xs(input, expected_output):
    output = apf.average_and_interpolate(input, N=len(expected_output))
    assert np.isclose(output.y, expected_output).all()


@pytest.mark.parametrize(
    "ys, xs, expected_output, N",
    [
        (
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            [1, 1, 1, 1, 1],
            5,
        ),
        (
            [[1, 1, 1, 1, 1], [1, 1, 1]],
            [[0, 1, 2, 3, 4], [0, 1, 2]],
            [1, 1, 1, 1, 1],
            5,
        ),
        (
            [[1, 1, 1, 1, 1], [1, 1, 1]],
            [[0, 1, 2, 3, 4], [2, 3, 4]],
            [1, 1, 1, 1, 1],
            5,
        ),
        (
            [[1, 1, 1, 1, 1], [3, 3, 3, 3, 3]],
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            [2, 2, 2, 2, 2],
            5,
        ),
        (
            [[1, 1, 1, 1, 1], [3, 3, 3]],
            [[0, 1, 2, 3, 4], [0, 1, 2]],
            [2, 2, 2, 1, 1],
            5,
        ),
        (
            [[1, 1, 1, 1, 1], [3, 3, 3]],
            [[0, 1, 2, 3, 4], [2, 3, 4]],
            [1, 1, 2, 2, 2],
            5,
        ),
        (
            [[1, 1, 1, 1, 1], [3, 3, 3, 3]],
            [[0, 1, 2, 3, 4], [2, 3, 4, 5]],
            [1, 1, 2, 2, 2, 3],
            6,
        ),
    ],
)
def test_average_list_with_xs(ys, xs, expected_output, N):
    output = apf.average_and_interpolate(ys=ys, xs=xs, N=N)
    assert np.isclose(output.y, expected_output).all()


def test_empty_average():
    N = 200
    avg = apf.average_and_interpolate([], [], N=N)

    assert avg.xs == []
    assert avg.ys == []
    assert (avg.y == np.zeros(N)).all()
    assert (avg.x == np.arange(N)).all()


@pytest.mark.parametrize(
    "length, num_beats",
    [(1, 1), (2, 2), (1, 0), (0, 1), (5, 10), (10, 5)],
)
def test_average_and_interpolate_same_length(length, num_beats):

    xs = [np.arange(length) for i in range(num_beats)]
    ys = [i * np.ones(length) for i in range(num_beats)]
    avg = apf.average_and_interpolate(ys, xs, N=length)

    x = np.arange(length) if len(xs) == 0 else xs[0]
    assert np.isclose(avg.x, x).all()

    y = np.zeros(length) if len(ys) == 0 else np.mean(ys, 0)
    assert np.isclose(avg.y, y).all()
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


def test_raises_Error_on_different_number_of_signals():
    with pytest.raises(apf.average.InvalidSubSignalError):
        apf.average_and_interpolate([[1], [1]], [[1]])


def test_raises_Error_on_different_signal_lengths():
    with pytest.raises(apf.average.InvalidSubSignalError):
        apf.average_and_interpolate([[1], [1, 2]], [[1], [2]])


@pytest.mark.parametrize("N", (2, 3, 4, 5, 6))
def test_create_longest_time_array(N):
    xs = [[0, 1, 2], [1, 2, 3]]
    x = apf.average.create_longest_time_array(xs, N)
    assert len(x) == N
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 3)
