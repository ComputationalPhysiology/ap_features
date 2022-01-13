import ap_features as apf
import numpy as np
import pytest


@pytest.mark.parametrize("spike_dur", [0, 1, 4, 8])
def test_find_spike_points(spike_dur):
    spike_pt = 2
    N = 10
    pacing = np.zeros(N)
    pacing[spike_pt + 1] = 1
    arr = np.arange(N)

    spike_points = apf.filters.find_spike_points(pacing, spike_dur)
    arr1 = np.delete(arr, spike_points)
    assert all(
        arr1
        == np.concatenate(
            (np.arange(spike_pt), np.arange(spike_pt + spike_dur, N)),
        ),
    )


def test_remove_points():
    x = np.arange(10)
    y = np.arange(10, 20)

    t_start = 5
    t_end = 7

    x_new, y_new = apf.filters.remove_points(x, y, t_start, t_end)

    assert np.isclose(x_new, np.arange(7)).all()
    assert np.isclose(y_new, [10, 11, 12, 13, 14, 18, 19]).all()


def test_filt():
    y = np.random.random(size=10)
    new_y = apf.filters.filt(y)
    assert y.size == new_y.size
    # Some values should be equal
    assert np.isclose(np.abs(y - new_y).min(), 0)


@pytest.mark.parametrize(
    "arr, x, expected_output",
    [
        ([1, 2, 3, 4, 5], 1, [1, 2, 3]),
        ([1, 1, 1], 1, [0, 1, 2]),
        ([0, 1, 1, 1], 1, [1, 2, 3]),
        ([], 1, []),
        ([1], 1, [0]),
    ],
)
def test_within_x_std(arr, x, expected_output):
    output = apf.filters.within_x_std(arr, x)
    assert len(output) == len(expected_output)
    assert np.isclose(output, expected_output).all()


def test_filter_signals_dict():
    data = {"apd30": [1, 2, 3, 4, 5], "length": [1, 1, 1, 3, 0]}

    output = apf.filters.filter_signals(data, x=1.0)
    assert output == [1, 2]


def test_filter_signals_list():
    data = [[1, 2, 3, 4, 5], [1, 1, 1, 3, 0]]

    output = apf.filters.filter_signals(data, x=1.0)
    assert output == [1, 2]


def test_filter_signales_raises_RuntimeError_on_unequal_lengths():
    data = {"apd30": [1, 2, 3, 4, 5], "length": [1, 1, 1, 0]}

    with pytest.raises(RuntimeError):
        apf.filters.filter_signals(data, x=1.0)
