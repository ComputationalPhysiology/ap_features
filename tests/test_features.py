import itertools as it
import os

import ap_features as apf
import ap_features as cost_terms
import numpy as np
import pytest


here = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "factor, backend",
    it.product(range(10, 95, 5), cost_terms.Backend),
)
def test_apds_triangle_signal(factor, backend, triangle_signal):
    x, y = triangle_signal

    apd = cost_terms.apd(V=y, t=x, factor=factor, backend=backend)
    assert abs(apd - 2 * factor) < 1e-10


@pytest.mark.parametrize(
    "factor_x, factor_y, backend",
    it.product(range(10, 95, 5), range(10, 95, 5), apf.Backend),
)
def test_apdxy_triangle_signal(factor_x, factor_y, backend, triangle_signal):
    x, y = triangle_signal

    apd = apf.apd_up_xy(y, x, factor_x, factor_y)

    if factor_x == factor_y:
        assert abs(apd) < 1e-12
    else:
        assert abs(apd - (factor_y - factor_x)) < 1e-10


def test_number_of_cost_terms():
    assert cost_terms.NUM_COST_TERMS == len(cost_terms.list_cost_function_terms())


def test_number_of_cost_terms_trace():
    assert cost_terms.NUM_COST_TERMS // 2 == len(
        cost_terms.list_cost_function_terms_trace(),
    )


def test_compare_python_matlab(synthetic_data):

    arr, t, expected_cost = synthetic_data
    cost = cost_terms.cost_terms(v=arr[0, :], ca=arr[1, :], t_v=t, t_ca=t)
    lst = cost_terms.list_cost_function_terms()

    i = 0
    for ri in expected_cost:

        if i in np.where(["APD_up" in item or "CaD_up" in item for item in lst])[0]:
            continue
        msg = f"{i}\tPython: {cost[i]}, Matlab {ri}" ""
        # print(msg)
        assert abs(cost[i] - ri) < 1e-10, msg
        i += 1


def test_compare_c_matlab(synthetic_data):

    arr, t, expected_cost = synthetic_data

    cost = apf.cost_terms(
        v=np.ascontiguousarray(arr[0, :]),
        ca=np.ascontiguousarray(arr[1, :]),
        t_v=t,
        t_ca=t,
    )

    lst = apf.list_cost_function_terms()
    i = 0
    for ri in expected_cost:

        if i in np.where(["APD_up" in item or "CaD_up" in item for item in lst])[0]:
            continue

        if i in np.where(["int_30" in item for item in lst])[0]:
            tol = 0.02
        else:
            tol = 1e-10

        assert abs((cost[i] - ri) / ri) < tol
        i += 1


@pytest.mark.parametrize(
    "y, x, expected_ttp",
    [
        ([0, 1, 0], [0, 1, 2], 1),
        ([0, 1, 2], [0, 1, 2], 2),
        ([2, 1, 0], [0, 1, 2], 0),
        ([], [], 0),
        ([0, 0, 0], [0, 1, 2], 0),
    ],
)
def test_time_to_peak_without_pacing(y, x, expected_ttp):
    assert apf.features.time_to_peak(y, x) == expected_ttp


@pytest.mark.parametrize(
    "y, x, p, expected_ttp",
    [
        ([0, 0, 0, 1], [0, 1, 2, 3], [0, 1, 0, 0], 2),
        ([0, 0, 1, 0], [0, 1, 2, 3], [0, 1, 0, 0], 1),
        ([0, 0, 1, 0], [0, 1, 2, 3], [0, 0, 0, 0], 2),
    ],
)
def test_time_to_peak_with_pacing(y, x, p, expected_ttp):
    assert apf.features.time_to_peak(y, x, pacing=p) == expected_ttp


@pytest.mark.parametrize("backend", ("c", "numba"))
def test_all_cost_terms(synthetic_data, backend):

    arr, t, expected_cost = synthetic_data
    arrs = np.expand_dims(arr, axis=0)
    cost = apf.all_cost_terms(arrs.T, t, backend=backend).squeeze()

    lst = apf.list_cost_function_terms()
    up_inds = np.where(["APD_up" in item or "CaD_up" in item for item in lst])[0]
    cost = np.delete(cost, up_inds)

    assert np.all(cost - expected_cost < 1e-10)


def test_cost_terms_trace(synthetic_data):
    arr, t, expected_cost = synthetic_data
    V = np.ascontiguousarray(arr[0, :])

    cost_terms_c = apf.cost_terms_trace(V, t, backend="c")
    cost_terms_py = apf.cost_terms_trace(V, t, backend="numba")

    lst = apf.list_cost_function_terms_trace()
    inds = np.where(["int_30" in item for item in lst])[0]
    x = np.delete(np.arange(len(lst)), inds)

    assert np.all(np.abs(cost_terms_c[x] - cost_terms_py[x]) < 1e-10)


@pytest.mark.parametrize("factor, backend", it.product((40, 60, 80), ("c", "numba")))
def test_apd_equivalence(factor, backend, synthetic_data):
    arr, t, expected_cost = synthetic_data
    V = np.ascontiguousarray(arr[0, :])
    apd_py = apf.apd(V=V, t=t, factor=factor, backend="python")
    apd_x = apf.apd(V=V, t=t, factor=factor, backend=backend)

    # We expect some difference here, but no more than 1ms
    assert abs(apd_x - apd_py) < 1


@pytest.mark.parametrize("factor", (40, 60, 80))
def test_apd_equivalence_c_numba(factor, synthetic_data):
    arr, t, expected_cost = synthetic_data
    V = np.ascontiguousarray(arr[0, :])
    apd_numba = apf.apd(V=V, t=t, factor=factor, backend="numba")
    apd_c = apf.apd(V=V, t=t, factor=factor, backend="c")

    assert abs(apd_c - apd_numba) < 1e-10


@pytest.mark.parametrize(
    "key, index0, index4",
    (("", "max", "APD10"), ("V", "V_max", "APD10"), ("Ca", "Ca_max", "CaD10")),
)
def test_list_cost_function_terms_trace(key, index0, index4):
    lst = apf.list_cost_function_terms_trace(key)
    assert lst[0] == index0
    assert lst[4] == index4


def test_list_cost_function_terms():
    lst = apf.list_cost_function_terms()
    assert lst[0] == "V_max"
    assert lst[apf.NUM_COST_TERMS // 2] == "Ca_max"


@pytest.mark.parametrize(
    "a, tau",
    [
        (0.1, 319.4724869136086),
        (0.3, 193.91125351445308),
        (0.5, 131.1866357539183),
        (0.7, 84.85669333330526),
        (0.9, 41.08344683897343),
    ],
)
def test_tau(a, tau, calcium_trace):
    t, y = calcium_trace
    assert np.isclose(apf.features.tau(t, y, a), tau)


@pytest.mark.parametrize(
    "p, int_p",
    [
        (10, 4.522984104069413),
        (30, 24.622790520357395),
        (50, 56.081175857097925),
        (70, 99.92180272118017),
        (90, 162.06786223150803),
    ],
)
def test_integrate_apd_use_spline(p, int_p, calcium_trace):
    t, y = calcium_trace
    assert np.isclose(apf.features.integrate_apd(y, t, p), int_p)


@pytest.mark.parametrize(
    "use_spline, tri",
    [(True, 156.22306861466004), (False, 151.51515151515153)],
)
def test_triangulation(use_spline, tri, calcium_trace):
    t, y = calcium_trace
    assert np.isclose(apf.features.triangulation(y, t, use_spline=use_spline), tri)


@pytest.mark.parametrize(
    "factor, use_spline",
    it.product(range(10, 90, 5), [True, False]),
)
def test_apd_coords(factor, use_spline, triangle_signal):
    x, y = triangle_signal
    apd_coords = apf.features.apd_coords(factor, y, x, use_spline=use_spline)
    # breakpoint()
    assert np.isclose(apd_coords.x1, 100 - factor, atol=1)
    assert np.isclose(apd_coords.x2, 100 + factor, atol=1)
    assert np.isclose(apd_coords.y1, 1 - factor / 100, atol=0.1)
    assert np.isclose(apd_coords.y2, 1 - factor / 100, atol=0.1)
    assert np.isclose(apd_coords.yth, 1 - factor / 100)


@pytest.mark.parametrize("factor", range(10, 90, 5))
def test_upstroke(factor, triangle_signal):
    x, y = triangle_signal
    assert np.isclose(apf.features.upstroke(x, y, a=factor / 100), factor)


def test_beating_frequency(multiple_beats):
    x, y = multiple_beats
    beats = apf.Beats(y, x).beats
    times = [beat.t for beat in beats]
    assert np.isclose(
        apf.features.beating_frequency(times),
        12.671594508975712,
        atol=0.2,
    )


def test_beating_frequency_from_peaks(multiple_beats):
    x, y = multiple_beats
    beats = apf.Beats(y, x).beats
    times = [beat.t for beat in beats]
    signals = [beat.y for beat in beats]
    assert np.isclose(
        apf.features.beating_frequency_from_peaks(signals, times),
        12.8,
        atol=0.2,
    ).all()


def test_max_relative_upstroke_velocity_sigmoid(calcium_trace):
    x, y = calcium_trace
    max_up = apf.features.max_relative_upstroke_velocity(x, y, sigmoid_fit=True)
    assert np.isclose(max_up.value, 0.03523266321708668)
    assert np.isclose(max_up.x0, 23.41210719885557)


def test_max_relative_upstroke_velocity_no_sigmoid(calcium_trace):
    x, y = calcium_trace
    max_up = apf.features.max_relative_upstroke_velocity(x, y, sigmoid_fit=False)
    assert np.isclose(max_up.value, 0.03773708905899542)
    assert np.isclose(max_up.index, 16)


def test_maximum_upstroke_velocity_use_spline(calcium_trace):
    x, y = calcium_trace
    max_up = apf.features.maximum_upstroke_velocity(y, t=x, use_spline=True)
    assert np.isclose(max_up, 0.037779410987573904)


def test_maximum_upstroke_velocity(calcium_trace):
    x, y = calcium_trace
    max_up = apf.features.maximum_upstroke_velocity(y, t=x, use_spline=False)
    assert np.isclose(max_up, 0.03282385532831371)


@pytest.mark.parametrize(
    "factor",
    range(10, 90, 5),
)
def test_apd_point_triangle_signal(factor, triangle_signal):
    x, y = triangle_signal
    x0, x1 = apf.features.apd_point(factor=factor, V=y, t=x, use_spline=True)
    assert np.isclose(x0, 100 - factor)
    assert np.isclose(x1, 100 + factor)
