import numpy as np


def list_cost_function_terms_trace(key=""):

    apd_key = "APD"
    if key.lower() == "ca":
        apd_key = "CaD"

    if key != "":
        key += "_"

    lst = (
        [f"{key}max", f"{key}min", f"{key}t_max", f"d{key}dt_max"]
        + [f"{apd_key}{apd}" for apd in np.arange(10, 95, 5, dtype=int)]
        + [
            f"{apd_key}_up_{x}{y}"
            for x in np.arange(20, 61, 20, dtype=int)
            for y in np.arange(x + 20, 81, 20, dtype=int)
        ]
        + [f"{key}int_30", f"{key}t_up", f"{key}t_down"]
    )
    return lst


def list_cost_function_terms():
    return list_cost_function_terms_trace("V") + list_cost_function_terms_trace("Ca")
