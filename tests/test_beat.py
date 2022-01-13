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


def test_beats_backgroud(real_trace):
    beats_no_backgroud = apf.Beats(
        real_trace.y,
        real_trace.t,
        real_trace.pacing,
        "none",
    )
    beats_full_backgroud = apf.Beats(
        real_trace.y,
        real_trace.t,
        real_trace.pacing,
        "full",
    )
    beats_subtract_backgroud = apf.Beats(
        real_trace.y,
        real_trace.t,
        real_trace.pacing,
        "subtract",
    )

    assert (
        beats_full_backgroud.y.max()
        < beats_subtract_backgroud.y.max()
        < beats_no_backgroud.y.max()
    )

    # These values should be close to zero
    assert abs(beats_full_backgroud.y.min()) < 1
    assert abs(beats_full_backgroud.y.max()) < 1
    assert abs(beats_subtract_backgroud.y.min()) < 1


def test_corrected_apd(real_beats):

    # Freqeuncy should be close to 1.5Hz
    assert abs(real_beats.beating_frequency - 1.5) < 0.01
    # The beat rate should be 60*1.5 = 90
    assert abs(real_beats.beat_rate - 90) < 1

    first_beat: apf.Beat = real_beats.beats[0]
    apd50 = first_beat.apd(50)
    capd50 = first_beat.capd(50)
    # Frequency is higher than 1 so capd should be larger than apd
    assert capd50 > apd50

    # If beat rate is 60 (i.e 1 Hz) then cAPD = APD
    apd50_2 = first_beat.capd(50, beat_rate=60)
    assert np.isclose(apd50, apd50_2)


def test_corrected_apd_with_no_parent_raises_RuntimeError(real_beats):
    first_beat: apf.Beat = real_beats.beats[0]
    clean_beat = apf.Beat(first_beat.y, first_beat.t)
    with pytest.raises(RuntimeError):
        clean_beat.capd(50)
    beat_rate = real_beats.beat_rates[0]
    assert np.isclose(first_beat.capd(50), clean_beat.capd(50, beat_rate=beat_rate))


def test_beats_slice(real_beats):
    # Slice 6 time steps (each time step is 10 ms)
    sliced_beats = real_beats.slice(1000, 1051)
    assert len(sliced_beats) == 6
    assert np.isclose(sliced_beats.t[0], 1000, atol=1)
    assert np.isclose(sliced_beats.t[-1], 1050, atol=1)


def test_beats_remove_spikes():
    spike_pt = 2
    spike_dur = 4
    N = 10
    pacing = np.zeros(N)
    pacing[spike_pt + 1] = 1

    trace = apf.Beats(np.arange(N), t=np.arange(N), pacing=pacing)
    new_trace = trace.remove_spikes(spike_dur)

    assert all(
        new_trace.y
        == np.concatenate(
            (np.arange(spike_pt), np.arange(spike_pt + spike_dur, N)),
        ),
    )

    assert len(new_trace) == N - spike_dur
    assert len(new_trace.y) == len(new_trace.t) == len(new_trace.pacing)


def test_beat_corrected_background(real_trace: apf.Beats):
    trace = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing)
    corrected1 = trace.correct_background(apf.BC.full)
    corrected2 = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing, apf.BC.full)

    assert corrected1 == corrected2
    assert np.isclose(corrected1.background, corrected2.background).all()


def test_beat_chop_data(real_trace):
    trace = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing)
    chopped_data = trace.chopped_data

    assert len(chopped_data.intervals) == 9
    assert len(chopped_data.data) == 9

    # First beat should be idendical
    y0 = chopped_data.data[0]
    i = next(i for i, p in enumerate(real_trace.pacing) if p > 0)
    assert np.isclose(chopped_data.data[0], real_trace.y[i - 2 : i - 2 + len(y0)]).all()


def test_beat_filter(real_trace):
    trace = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing)
    filtered_trace = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing).filter(
        kernel_size=3,
    )
    diff = np.abs(trace.y - filtered_trace.y)
    assert np.isclose(np.min(diff), 0)
    assert np.max(diff) < np.abs(np.diff(trace.y)).max()


def test_beat_remove_points(real_trace):
    trace = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing)
    t_start = 1000
    t_end = 2000
    new_trace = trace.remove_points(t_start, t_end)
    t_diff = t_end - t_start
    dt = 10
    assert abs(abs(new_trace.t[-1] - trace.t[-1]) - t_diff) < dt + 1
    assert abs(len(trace) - len(new_trace)) <= (t_diff / dt) + 1


def test_chopped_data_to_beats(real_trace):
    trace = apf.Beats(real_trace.y, real_trace.t, real_trace.pacing)
    chopped_data = trace.chopped_data
    beats = apf.beat.chopped_data_to_beats(chopped_data, parent=trace)

    assert len(beats) == len(chopped_data.data)
    first_beat = beats[0]
    assert first_beat.parent == trace
    assert (first_beat.y == chopped_data.data[0]).all()
    assert (first_beat.t == chopped_data.times[0]).all()
    assert (first_beat.pacing == chopped_data.pacing[0]).all()


def test_ensure_time_unit(real_beats):
    beat: apf.Beat = real_beats.beats[0]
    time_orig = np.copy(beat.t)
    assert beat.time_unit == "ms"

    beat.ensure_time_unit("ms")
    assert beat.time_unit == "ms"
    assert (np.isclose(time_orig, beat.t)).all()

    beat.ensure_time_unit("s")
    assert beat.time_unit == "s"
    assert (np.isclose(time_orig, 1000.0 * beat.t)).all()

    beat.ensure_time_unit("ms")
    assert beat.time_unit == "ms"
    assert (np.isclose(time_orig, beat.t)).all()


@pytest.mark.parametrize(
    "corrected_apd, expected_output",
    [
        (True, (-34.613263202359164, 386.6834901571444)),
        (False, (-27.795433438981334, 337.76067618613905)),
    ],
)
def test_apd_slope(corrected_apd, expected_output, real_beats):
    output = real_beats.apd_slope(80, corrected_apd=corrected_apd)
    assert np.isclose(output, expected_output).all()


def test_detect_ead_no_ead(real_beats):
    beat: apf.Beat = real_beats.beats[0]
    has_ead, index = beat.detect_ead()
    assert has_ead is False
    assert index is None


def test_detect_ead_with_ead(real_beats):
    beat: apf.Beat = real_beats.beats[0].copy()

    # Add artifial EAD
    bump = np.zeros_like(beat.t)
    bump[25:30] = 3
    beat.y[:] += bump

    has_ead, index = beat.detect_ead()
    assert has_ead is True
    assert index == 26
