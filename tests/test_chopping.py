import ap_features as apf
import numpy as np
import pytest
import scipy


@pytest.fixture(params=["with_pacing", "without_pacing"])
def chopped_data(request):
    N = 700
    x = np.linspace(0, 7.0, N)
    time = x * 1000
    alpha = 0.74

    average = np.sin(2 * np.pi * (x + alpha))

    pacing = np.zeros(len(x))
    for r in range(8):
        pacing[1 + 100 * r : 100 * r + 10] = 1
    kwargs = dict(data=average, time=time, extend_front=0)
    if request.param == "with_pacing":
        kwargs["pacing"] = pacing
    else:
        kwargs["extend_front"] = 250

    kwargs["use_pacing_info"] = request.param == "with_pacing"

    yield apf.chopping.chop_data(**kwargs), kwargs


def test_chop_data(chopped_data):
    chopped_data, kwargs = chopped_data
    # assert len(chopped_data.data) == 7
    print([len(c) for c in chopped_data.data])
    assert chopped_data.parameters["use_pacing_info"] is kwargs["use_pacing_info"]
    if not kwargs["use_pacing_info"]:
        assert chopped_data.parameters["extend_front"] == kwargs["extend_front"]

    # fig, ax = plt.subplots()
    # for t, c in zip(chopped_data.times, chopped_data.data):
    #     ax.plot(t, c)
    # plt.show()
    N = min([len(d) for d in chopped_data.data])
    data = np.array([d[:N] for d in chopped_data.data])
    q = scipy.spatial.distance.pdist(data, "euclidean") / max(
        [np.linalg.norm(d) for d in data],
    )
    assert all(q < 0.1)

    times = np.array([t[:N] for t in chopped_data.times])
    assert all(scipy.spatial.distance.pdist([t - t[0] for t in times]) < 1e-10)


@pytest.mark.parametrize(
    "starts, ends, extend_front, extend_end, expected_intervals",
    [
        (
            [69.40241445, 1628.85895293, 3074.24891969, 4641.53700245],
            [811.77991702, 2380.96124084, 3826.5365545],
            None,
            None,
            [
                (0, 1158.423756445),
                (1282.2151135, 2727.60508026),
                (2727.60508026, 4173.18039393),
            ],
        ),
        (
            [69.40241445, 1628.85895293, 3074.24891969, 4641.53700245],
            [811.77991702, 2380.96124084, 3826.5365545],
            0,
            0,
            [
                (69.40241445, 811.77991702),
                (1628.85895293, 2380.96124084),
                (3074.24891969, 3826.5365545),
            ],
        ),
    ],
)
def test_filter_start_ends_in_chopping(
    starts,
    ends,
    extend_front,
    extend_end,
    expected_intervals,
):
    intervals = apf.chopping.filter_start_ends_in_chopping(
        starts,
        ends,
        extend_front,
        extend_end,
    )
    assert all([np.isclose(i, e).all() for i, e in zip(intervals, expected_intervals)])


@pytest.mark.parametrize(
    "starts, ends",
    [
        ([], [1.0]),
        ([1.0], []),
        ([], []),
        ([2.0], [1.0]),
    ],
)
def test_filter_start_ends_in_chopping_raises_on_empty(starts, ends):
    with pytest.raises(apf.chopping.EmptyChoppingError):
        apf.chopping.filter_start_ends_in_chopping(starts, ends)


@pytest.mark.parametrize(
    "extend, intervals, default, value",
    [
        (None, [], 200, 200),
        (None, [(1, 1)], 200, 200),
        (100, [], 200, 100),
        (None, [(100, 200), (300, 400), (500, 600)], 200, 50),
    ],
)
def test_get_extend_value(extend, intervals, default, value):
    assert np.isclose(
        apf.chopping.get_extend_value(extend, intervals, default),
        value,
    )


@pytest.mark.parametrize(
    "starts, ends, expected_intervals",
    [
        ([1, 3], [2, 4], [(1, 2), (3, 4)]),
        ([1, 3, 5], [2, 4, 6], [(1, 2), (3, 4), (5, 6)]),
        ([1, 3, 5], [2, 4], [(1, 2), (3, 4)]),
        ([1, 3], [2, 4, 6], [(1, 2), (3, 4)]),
        ([3, 5], [2, 4, 6], [(3, 4), (5, 6)]),
        ([1, 3, 5], [4, 6], [(1, 4), (3, 6)]),
    ],
)
def test_create_interval(starts, ends, expected_intervals):
    intervals = apf.chopping.create_intervals(starts, ends)
    assert intervals == expected_intervals


def test_check_interval_raises_EmptyChoppingError():
    with pytest.raises(apf.chopping.EmptyChoppingError):
        apf.chopping.check_intervals([])


def test_check_interval_raises_InvalidChoppingError():
    with pytest.raises(apf.chopping.InvalidChoppingError):
        apf.chopping.check_intervals([(1, 2), (4, 3)])
