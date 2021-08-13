#!/usr/bin/env python
"""The setup script."""
import sys

from setuptools import Extension
from setuptools import setup


def check_for_openmp():
    import glob
    import os
    import subprocess
    import tempfile
    from distutils.ccompiler import new_compiler
    from distutils.errors import CompileError, LinkError

    CCODE = """
    #include <omp.h>
    #include <stdio.h>

    int main(void) {
        #pragma omp parallel num_threads(2)
        printf("nthreads=%d\\n", omp_get_num_threads());

        return 0;
    }
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        start_dir = os.path.abspath(".")

        try:
            os.chdir(tmp_dir)

            filename = "test_openmp.c"
            # Write test program
            with open(filename, "w") as f:
                f.write(CCODE)

            os.mkdir("objects")

            link_flags = get_openmp_link_args()
            compile_flags = get_openmp_compile_args()
            ccompiler = new_compiler()
            # Compile, test program
            ccompiler.compile(
                [filename],
                output_dir="objects",
                extra_postargs=compile_flags,
            )

            # Link test program
            objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))
            ccompiler.link_executable(
                objects,
                filename.split(".")[0],
                extra_postargs=link_flags,
            )

            # Run test program
            output = subprocess.check_output(f'./{filename.split(".")[0]}')
            output = output.decode(sys.stdout.encoding or "utf-8").splitlines()

            if "nthreads=" in output[0]:
                nthreads = int(output[0].strip().split("=")[1])
                if len(output) == nthreads:
                    is_openmp_supported = True
                else:
                    print(
                        "Unexpected number of lines from output of test OpenMP "
                        "program (output was {0})".format(output),
                    )
                    is_openmp_supported = False
            else:
                print(
                    "Unexpected output from test OpenMP "
                    "program (output was {0})".format(output),
                )
                is_openmp_supported = False
        except (CompileError, LinkError, subprocess.CalledProcessError):
            is_openmp_supported = False

        finally:
            os.chdir(start_dir)

    return is_openmp_supported


def get_openmp_compile_args():
    extra_compile_args = ["-fopenmp"]
    if sys.platform == "darwin":
        extra_compile_args = ["-Xclang", "-fopenmp"]
    elif sys.platform == "win32":
        extra_compile_args = ["-openmp"]
    return extra_compile_args


def get_openmp_link_args():
    extra_link_args = ["-fopenmp"]
    if sys.platform == "darwin":
        extra_link_args = ["-lomp"]
    elif sys.platform == "win32":
        extra_link_args = []
    return extra_link_args


def collect_c_functions(source_files):
    import re

    cdecl_re = re.compile(r"(\S+)\s+CDECL\s+(\w+)")

    def collect_c_functions_file(source_file):
        with open(source_file, "r") as f:
            code = f.read()

        m_list = cdecl_re.findall(code)
        function_names = [m[1] for m in m_list]
        return function_names

    all_function_names = []
    for source_file in source_files:
        all_function_names.extend(collect_c_functions_file(source_file))

    return all_function_names


def get_extension():
    import os

    extra_compile_args = []
    extra_link_args = []
    disable_openmp = "DISABLE_OPENMP" in os.environ
    if not disable_openmp and check_for_openmp():
        extra_compile_args = get_openmp_compile_args()
        extra_link_args = get_openmp_link_args()

    sources = ["src/c/cost_terms.c"]
    c_functions = collect_c_functions(sources)

    return Extension(
        "ap_features.libcost_terms",
        sources,
        language="c",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        export_symbols=c_functions,
    )


requirements = ["numpy", "numba", "tqdm"]

setup(
    ext_modules=[get_extension()],
    version="2021.0.1",
)
