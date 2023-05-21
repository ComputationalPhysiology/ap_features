__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "2023.1.3"

import logging as _logging

from . import (
    background,
    chopping,
    features,
    average,
    filters,
    utils,
    _numba,
    plot,
    testing,
)

from ._numba import transpose_trace_array
from .beat import Beat, Beats, Trace, BeatCollection, StateCollection, State
from .features import all_cost_terms, apd, apd_up_xy, cost_terms, cost_terms_trace
from .utils import (
    Backend,
    list_cost_function_terms,
    list_cost_function_terms_trace,
    NUM_COST_TERMS,
)
from .average import average_and_interpolate
from .background import BackgroundCorrection as BC


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
