import ap_features as apf
import dask.array as da
import h5py
import numpy as np
import pytest
from ap_features.beat import Beat


def handle_request_param(request, t, y, cls):
    if request.param == "numpy":
        yield cls(y, t)
    elif request.param == "dask":
        y_ = da.from_array(y)
        t_ = da.from_array(t)
        yield cls(y_, t_)
    else:
        # Create an in memory file
        fp = h5py.File(name=cls.__name__, mode="w", driver="core", backing_store=False)
        y_ = fp.create_dataset("y", data=y)
        t_ = fp.create_dataset("t", data=t)
        yield cls(y_, t_)
        fp.close()


@pytest.fixture(params=["numpy", "dask", "h5py"])
def trace(request, single_beat):
    t, y = single_beat
    yield from handle_request_param(request, t, y, apf.Trace)


@pytest.fixture(params=["numpy", "dask", "h5py"])
def beat(request, single_beat):
    t, y = single_beat
    yield from handle_request_param(request, t, y, apf.Beat)


@pytest.fixture(params=["numpy", "dask", "h5py"])
def state(request, single_beat, NUM_STATES):
    t, y = single_beat
    ys = np.repeat(y, NUM_STATES).reshape(-1, NUM_STATES)
    yield from handle_request_param(request, t, ys, apf.State)


@pytest.fixture(params=["numpy", "dask", "h5py"])
def beats(request, multiple_beats):
    t, y = multiple_beats
    yield from handle_request_param(request, t, y, apf.Beats)


@pytest.fixture(params=["numpy", "dask", "h5py"])
def beatcollection(request, single_beat_collection):
    t, y = single_beat_collection
    yield from handle_request_param(request, t, y, apf.BeatCollection)


@pytest.fixture(params=["numpy", "dask", "h5py"])
def statecollection(request, state_collection_data):
    t, y = state_collection_data
    yield from handle_request_param(request, t, y, apf.StateCollection)


def test_trace(single_beat, trace):
    t, y = single_beat

    assert (abs(trace.t - t) < 1e-12).all()
    assert (abs(trace.y - y) < 1e-12).all()
    assert (abs(trace.pacing - 0) < 1e-12).all()


def test_beat(single_beat, beat):
    assert beat.is_valid()
    t, y = single_beat
    assert (abs(beat.t - t) < 1e-12).all()
    assert (abs(beat.y - y) < 1e-12).all()
    assert (abs(beat.pacing - 0) < 1e-12).all()


def test_beats(beats, multiple_beats):
    t, y = multiple_beats
    assert (abs(beats.t - t) < 1e-12).all()
    assert (abs(beats.y - y) < 1e-12).all()
    assert (abs(beats.pacing - 0) < 1e-12).all()


def test_beat_equality(beat):
    new_beat = Beat(beat.y.copy(), beat.t.copy())
    assert beat == new_beat
    new_beat.y[0] += 1
    assert beat != new_beat


def test_remove_bad_indices():
    bad_indices = {2, 3}
    feature_list = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    new_feature_list = apf.beat.remove_bad_indices(feature_list, bad_indices)
    assert new_feature_list == [[1, 2], [1, 2], [1, 2]]


def test_filter_beats(beats):
    filtered_beats = apf.beat.filter_beats(beats.beats, ["apd30", "length", "apd80"])
    # FIXME: mock calls so that we know which beats that are included
    assert len(filtered_beats) > 0


def test_filter_beats_no_filter(beats):
    filtered_beats = apf.beat.filter_beats(beats.beats, [])
    assert all([fb == b for fb, b in zip(filtered_beats, beats.beats)])


@pytest.mark.parametrize("filters", [None, {}, ["apd30", "length"]])
def test_average_beat(beats, filters):
    # Just a smoke test
    avg = beats.average_beat(filters=filters)
    assert isinstance(avg, Beat)


def test_beatcollection(beatcollection, single_beat_collection, NUM_TRACES):
    t, y = single_beat_collection
    assert (abs(beatcollection.t - t) < 1e-12).all()
    assert (abs(beatcollection.y - y) < 1e-12).all()
    assert (abs(beatcollection.pacing - 0) < 1e-12).all()
    assert beatcollection.num_traces == NUM_TRACES


def test_statecollection(
    statecollection,
    state_collection_data,
    NUM_TRACES,
    NUM_STATES,
):
    t, y = state_collection_data
    assert (abs(statecollection.t - t) < 1e-12).all()
    assert (abs(statecollection.y - y) < 1e-12).all()
    assert (abs(statecollection.pacing) - 0 < 1e-12).all()
    assert statecollection.num_traces == NUM_TRACES
    assert statecollection.num_states == NUM_STATES


def test_state_collection_cost_terms(statecollection, state, NUM_TRACES):
    c1 = state.cost_terms
    c2 = statecollection.cost_terms
    for i in range(NUM_TRACES):
        assert (abs(c2[i, :] - c1) < 1e-12).all()
