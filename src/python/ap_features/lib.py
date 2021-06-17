"""This module contains the library utilities and the bindings
to all the C functions
"""
import ctypes
import logging
import os
import time
from ctypes import c_double
from ctypes import c_int
from ctypes import c_long
from ctypes import c_uint8
from ctypes import c_void_p
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
HERE = Path(__file__).absolute().parent


def load_library(name: str) -> ctypes.CDLL:
    """Load ctypes library. Search in the current
    directory for shared library files

    Parameters
    ----------
    name : str
        Name of shared library

    Returns
    -------
    ctypes.CDLL
        The shared library

    Raises
    ------
    FileNotFoundError
        If it cannot find the library
    """

    try:
        libname = next(f.name for f in HERE.iterdir() if f.name.startswith(name))
    except StopIteration:
        raise FileNotFoundError(f"Could not find shared library for {name}")

    lib = np.ctypeslib.load_library(libname, HERE)

    lib_path = lib._name
    lib_mtime_float = os.path.getmtime(lib_path)
    lib_mtime_struct = time.localtime(lib_mtime_float)
    lib_mtime_str = time.asctime(lib_mtime_struct)
    logger.debug(f"Loading library '{lib_path}' last modified {lib_mtime_str}")
    return lib


lib = load_library("libcost_terms")

uint8_array = np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1, flags="contiguous")
float64_array = np.ctypeslib.ndpointer(dtype=c_double, ndim=1, flags="contiguous")
float64_array_2d = np.ctypeslib.ndpointer(dtype=c_double, ndim=2, flags="contiguous")
float64_array_3d = np.ctypeslib.ndpointer(dtype=c_double, ndim=3, flags="contiguous")

lib.apd.argtypes = [float64_array, float64_array, c_int, c_int, float64_array]
lib.apd.restype = c_double

lib.apd_up_xy.argtypes = [
    float64_array,
    float64_array,
    c_int,
    c_int,
    c_int,
    float64_array,
]
lib.apd_up_xy.restype = c_double

lib.cost_terms_trace.argtypes = [float64_array, float64_array, float64_array, c_int]
lib.cost_terms_trace.restype = None

lib.get_num_cost_terms.argtypes = []
lib.get_num_cost_terms.restype = c_int

lib.full_cost_terms.argtypes = [
    float64_array,
    float64_array,
    float64_array,
    float64_array,
    c_int,
]
lib.full_cost_terms.restype = None

lib.all_cost_terms.argtypes = [
    float64_array_2d,
    float64_array_3d,
    float64_array,
    uint8_array,
    c_long,
    c_long,
    c_void_p,
]
lib.all_cost_terms.restype = None
