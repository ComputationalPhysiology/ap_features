import itertools as it
import os

import ap_features as apf
import ap_features as cost_terms
import numpy as np
import pytest

here = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def triangle_signal():
    x = np.arange(301, dtype=float)

    y = np.zeros(301, dtype=float)
    y[:101] = np.linspace(0, 1, 101)
    y[101:201] = np.linspace(1, 0, 101)[1:]
    return (x, y)


@pytest.fixture(scope="session")
def synthetic_data(num_parameter_sets=1):
    data = np.load(os.path.join(here, "data.npy"), allow_pickle=True).item()
    cost = np.load(os.path.join(here, "cost_terms.npy"))
    trace = np.array([data["v"], data["ca"]])
    return trace, data["t"], cost


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
    output = apf.features.within_x_std(arr, x)
    assert len(output) == len(expected_output)
    assert np.isclose(output, expected_output).all()


def test_filter_signals_dict():
    data = {"apd30": [1, 2, 3, 4, 5], "length": [1, 1, 1, 3, 0]}

    output = apf.features.filter_signals(data, x=1.0)
    assert output == [1, 2]


def test_filter_signals_list():
    data = [[1, 2, 3, 4, 5], [1, 1, 1, 3, 0]]

    output = apf.features.filter_signals(data, x=1.0)
    assert output == [1, 2]


def test_filter_signales_raises_RuntimeError_on_unequal_lengths():
    data = {"apd30": [1, 2, 3, 4, 5], "length": [1, 1, 1, 0]}

    with pytest.raises(RuntimeError):
        apf.features.filter_signals(data, x=1.0)


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


def test_tau():
    pass
