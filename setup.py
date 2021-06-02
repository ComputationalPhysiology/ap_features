#!/usr/bin/env python

"""The setup script."""
import sys

from setuptools import Extension, setup


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
        #pragma omp parallel
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
                [filename], output_dir="objects", extra_postargs=compile_flags
            )

            # Link test program
            objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))
            ccompiler.link_executable(
                objects, filename.split(".")[0], extra_postargs=link_flags
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
                        "program (output was {0})".format(output)
                    )
                    is_openmp_supported = False
            else:
                print(
                    "Unexpected output from test OpenMP "
                    "program (output was {0})".format(output)
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
    return extra_compile_args


def get_openmp_link_args():
    extra_link_args = ["-fopenmp"]
    if sys.platform == "darwin":
        extra_link_args = ["-lomp"]
    return extra_link_args


def get_extension():
    extra_compile_args = []
    extra_link_args = []
    if check_for_openmp():
        extra_compile_args = get_openmp_compile_args()
        extra_link_args = get_openmp_link_args()

    return Extension(
        "ap_features.cost_terms",
        ["src/c/cost_terms.c"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


requirements = ["numpy", "numba", "tqdm"]

setup(
    ext_modules=[get_extension()],
    version="0.1.2",
)
