import ap_features as apf
import dask.array as da
import h5py
import numpy as np
import pytest


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


@pytest.fixture(scope="session", params=["numpy", "dask", "h5py"])
def trace(request, single_beat):
    t, y = single_beat
    yield from handle_request_param(request, t, y, apf.Trace)


@pytest.fixture(scope="session", params=["numpy", "dask", "h5py"])
def beat(request, single_beat):
    t, y = single_beat
    yield from handle_request_param(request, t, y, apf.Beat)


@pytest.fixture(scope="session", params=["numpy", "dask", "h5py"])
def state(request, single_beat, NUM_STATES):
    t, y = single_beat
    ys = np.repeat(y, NUM_STATES).reshape(-1, NUM_STATES)
    yield from handle_request_param(request, t, ys, apf.State)


@pytest.fixture(scope="session", params=["numpy", "dask", "h5py"])
def beatseries(request, multiple_beats):
    t, y = multiple_beats
    yield from handle_request_param(request, t, y, apf.BeatSeries)


@pytest.fixture(scope="session", params=["numpy", "dask", "h5py"])
def beatcollection(request, single_beat_collection):
    t, y = single_beat_collection
    yield from handle_request_param(request, t, y, apf.BeatCollection)


@pytest.fixture(scope="session", params=["numpy", "dask", "h5py"])
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


def test_beatseries(beatseries, multiple_beats):
    t, y = multiple_beats
    assert (abs(beatseries.t - t) < 1e-12).all()
    assert (abs(beatseries.y - y) < 1e-12).all()
    assert (abs(beatseries.pacing - 0) < 1e-12).all()


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
