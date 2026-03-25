__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "2023.1.3"

import logging as _logging

from . import (
    _numba,
    average,
    background,
    chopping,
    features,
    filters,
    plot,
    testing,
    utils,
)
from ._numba import transpose_trace_array
from .average import average_and_interpolate
from .background import BackgroundCorrection as BC
from .beat import Beat, BeatCollection, Beats, State, StateCollection, Trace
from .features import all_cost_terms, apd, apd_up_xy, cost_terms, cost_terms_trace
from .utils import (
    NUM_COST_TERMS,
    Backend,
    list_cost_function_terms,
    list_cost_function_terms_trace,
)


def set_log_level(level=_logging.INFO):
    for logger in [
        _numba.logger,
        background.logger,
        chopping.logger,
        features.logger,
        filters.logger,
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
    "filters",
    "BC",
    "plot",
    "testing",
]
