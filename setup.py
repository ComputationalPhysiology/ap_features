#!/usr/bin/env python

"""The setup script."""
import sys

from setuptools import Extension, setup

extra_compile_args = ["-fopenmp"]
if sys.platform == "darwin":
    extra_compile_args = ["-Xclang", "-fopenmp"]


requirements = ["numpy", "numba", "tqdm"]

setup(
    ext_modules=[
        Extension(
            "ap_features.cost_terms",
            ["src/cost_terms.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=["-lgomp"],
        )
    ],
    version="0.1.0",
)
