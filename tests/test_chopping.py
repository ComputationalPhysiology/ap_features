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
    assert chopped_data.parameters.use_pacing_info is kwargs["use_pacing_info"]

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
