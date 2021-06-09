import pytest

from ap_features import Beat, Trace


@pytest.fixture(scope="session")
def trace(single_beat):
    t, y = single_beat
    return Trace(t, y)


@pytest.fixture(scope="session")
def beat(single_beat):
    t, y = single_beat
    return Beat(t, y)


def test_trace(single_beat, trace):
    t, y = single_beat
    assert (trace.t == t).all()
    assert (trace.y == y).all()
    assert (trace.pacing == 0).all()


def test_beat(single_beat, beat):
    assert beat.is_valid()
    t, y = single_beat
    assert (beat.t == t).all()
    assert (beat.y == y).all()
    assert (beat.pacing == 0).all()
