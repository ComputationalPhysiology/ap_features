import itertools as it
import os

import numpy as np
import pytest

from ap_features import ap_features as cost_terms

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
    "factor, method", it.product(range(10, 95, 5), ["apd_c", "compute_APD"])
)
def test_apds_triangle_signal(factor, method, triangle_signal):
    x, y = triangle_signal
    func = getattr(cost_terms, method)
    apd = func(y, x, factor)
    assert abs(apd - 2 * factor) < 1e-10


@pytest.mark.parametrize(
    "factor_x, factor_y, method",
    it.product(range(10, 95, 5), range(10, 95, 5), ["apd_up_xy_c", "compute_APDUpxy"]),
)
def test_apdxy_triangle_signal(factor_x, factor_y, method, triangle_signal):
    x, y = triangle_signal
    func = getattr(cost_terms, method)

    apd = func(y, x, factor_x, factor_y)

    if factor_x == factor_y:
        assert abs(apd) < 1e-12
    elif factor_x > factor_y:
        assert np.isinf(apd)
    else:
        assert abs(apd - (factor_y - factor_x)) < 1e-10


def test_number_of_cost_terms():
    assert cost_terms.NUM_COST_TERMS == len(cost_terms.list_cost_function_terms())


def test_number_of_cost_terms_trace():
    assert cost_terms.NUM_COST_TERMS // 2 == len(
        cost_terms.list_cost_function_terms_trace()
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

    cost = cost_terms.cost_terms_c(
        v=np.ascontiguousarray(arr[0, :]),
        ca=np.ascontiguousarray(arr[1, :]),
        t_v=t,
        t_ca=t,
    )

    lst = cost_terms.list_cost_function_terms()
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


def test_all_cost_terms(synthetic_data):

    arr, t, expected_cost = synthetic_data
    arrs = np.expand_dims(arr, axis=0)
    cost = cost_terms.all_cost_terms(arrs.T, t).squeeze()

    lst = cost_terms.list_cost_function_terms()
    up_inds = np.where(["APD_up" in item or "CaD_up" in item for item in lst])[0]
    cost = np.delete(cost, up_inds)

    assert np.all(cost - expected_cost < 1e-10)


def test_all_cost_terms_c(synthetic_data):

    arr, t, expected_cost = synthetic_data
    arrs = np.expand_dims(arr, axis=0)
    cost = cost_terms.all_cost_terms_c(np.ascontiguousarray(arrs.T), t)

    lst = cost_terms.list_cost_function_terms()
    up_inds = np.where(["APD_up" in item or "CaD_up" in item for item in lst])[0]
    cost = np.delete(cost, up_inds)

    # We take out the int_30 terms because we use
    # different integration rules
    x = np.array([xi for xi in range(48) if xi not in [21, 45]])
    tol = np.ones(len(x)) * 1e-10

    assert np.all(np.abs(cost[x] - expected_cost[x]) < tol)


def test_cost_terms_trace(synthetic_data):
    arr, t, expected_cost = synthetic_data
    V = np.ascontiguousarray(arr[0, :])

    cost_terms_c = cost_terms.cost_terms_trace_c(V, t)
    cost_terms_py = cost_terms.cost_terms_trace(V, t)

    lst = cost_terms.list_cost_function_terms_trace()
    inds = np.where(["int_30" in item for item in lst])[0]
    x = np.delete(np.arange(len(lst)), inds)

    assert np.all(np.abs(cost_terms_c[x] - cost_terms_py[x]) < 1e-10)


@pytest.mark.parametrize("factor", (40, 60, 80))
def test_apd_equivalence(factor, synthetic_data):
    arr, t, expected_cost = synthetic_data
    V = np.ascontiguousarray(arr[0, :])
    apd_c = cost_terms.apd_c(V, t, factor)
    apd_py = cost_terms.compute_APD(V, t, factor)
    assert abs(apd_c - apd_py) < 1e-10


@pytest.mark.parametrize(
    "key, index0, index4",
    (("", "max", "APD10"), ("V", "V_max", "APD10"), ("Ca", "Ca_max", "CaD10")),
)
def test_list_cost_function_terms_trace(key, index0, index4):
    lst = cost_terms.list_cost_function_terms_trace(key)
    assert lst[0] == index0
    assert lst[4] == index4


def test_list_cost_function_terms():
    lst = cost_terms.list_cost_function_terms()
    assert lst[0] == "V_max"
    assert lst[cost_terms.NUM_COST_TERMS // 2] == "Ca_max"
