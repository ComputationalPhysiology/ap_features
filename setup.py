#!/usr/bin/env python

"""The setup script."""
import sys
from distutils.core import Extension, setup

from setuptools import find_packages

NAME = "ap_features"


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()


def get_install_extras_require():
    extras_require = {
        "dev": ["flake8", "isort", "mypy", "ipython", "pdbpp", "pytest", "pytest-cov"],
    }
    # Add automatically the 'all' target
    extras_require.update(
        {"all": [val for values in extras_require.values() for val in values]}
    )
    return extras_require


extra_compile_args = ["-fopenmp"]
if sys.platform == "darwin":
    extra_compile_args = ["-Xclang", "-fopenmp"]


requirements = ["numpy", "numba", "tqdm"]

setup(
    author="Henrik Finsberg",
    author_email="henriknf@simula.no",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    description="Package to compute features of traces from action potential models",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    extras_require=get_install_extras_require(),
    keywords="ap_features",
    name=NAME,
    packages=find_packages(include=["ap_features", "ap_features.*"]),
    test_suite="tests",
    url="https://github.com/finsberg/ap_features",
    ext_modules=[
        Extension(
            "ap_features.cost_terms",
            ["src/cost_terms.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=["-lgomp"],
        ),
    ],
    version="0.1.0",
    project_urls={
        "Documentation": "https://ap-features.readthedocs.io.",
        "Source": "https://github.com/finsberg/ap_features",
    },
    zip_safe=False,
)
