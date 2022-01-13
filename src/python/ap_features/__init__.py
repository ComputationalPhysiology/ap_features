__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "2022.0.0"

import logging as _logging

from . import (
    background,
    chopping,
    features,
    average,
    filters,
    lib,
    utils,
    _c,
    _numba,
    plot,
)
from ._c import NUM_COST_TERMS
from ._numba import transpose_trace_array
from .beat import Beat, Beats, Trace, BeatCollection, StateCollection, State
from .features import all_cost_terms, apd, apd_up_xy, cost_terms, cost_terms_trace
from .utils import Backend, list_cost_function_terms, list_cost_function_terms_trace
from .average import average_and_interpolate, average_list
from .background import BackgroundCorrection as BC


def set_log_level(level=_logging.INFO):
    for logger in [
        _c.logger,
        _numba.logger,
        background.logger,
        chopping.logger,
        features.logger,
        filters.logger,
        lib.logger,
        utils.logger,
    ]:
        logger.setLevel(level)


set_log_level()

__all__ = [
    "features",
    "apd",
    "Backend",
    "apd_up_xy",
    "cost_terms",
    "cost_terms_trace",
    "all_cost_terms",
    "peak_and_repolarization",
    "transpose_trace_array",
    "list_cost_function_terms_trace",
    "list_cost_function_terms",
    "NUM_COST_TERMS",
    "Beat",
    "Trace",
    "Beats",
    "background",
    "chopping",
    "BeatCollection",
    "StateCollection",
    "State",
    "average",
    "average_and_interpolate",
    "average_list",
    "filters",
    "BC",
    "plot",
]
