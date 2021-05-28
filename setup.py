#!/usr/bin/env python

"""The setup script."""
import os
import platform
import re
import subprocess
import sys
from distutils.spawn import find_executable
from distutils.version import LooseVersion

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        cmake_args = ["-DCMAKE_INSTALL_PREFIX=" + sys.prefix]

        build_args = []

        env = os.environ.copy()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake = find_executable("cmake")

        subprocess.check_call(
            [cmake, ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--install", "."] + build_args, cwd=self.build_temp
        )


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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Package to compute features of traces from action potential models",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    extras_require=get_install_extras_require(),
    keywords="ap_features",
    name="ap_features",
    packages=find_packages(include=["ap_features", "ap_features.*"]),
    test_suite="tests",
    url="https://github.com/finsberg/ap_features",
    ext_modules=[CMakeExtension("cmake_example", "src")],
    cmdclass=dict(build_ext=CMakeBuild),
    version="0.1.0",
    project_urls={
        "Documentation": "https://ap-features.readthedocs.io.",
        "Source": "https://github.com/finsberg/ap_features",
    },
    zip_safe=False,
)
