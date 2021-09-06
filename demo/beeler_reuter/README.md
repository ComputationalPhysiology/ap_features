# The Beeler Reuter model

In this demo you can try out the [Beeler Reuter model](https://models.cellml.org/e/1).
There are two versions in this folder, one implemented in pure python (`demo_python.py`) and one implemented using C with python bindings (`demo_c.py`). In order to run `demo_c.py` you also need
[gotran](https://pypi.org/project/gotran/).


## Code generation
The code in this repo is generated from the model in the cellML model repository using [gotran](https://pypi.org/project/gotran/), i.e first converting the `.cellml` file to a `.ode` using `cellml2gotran` and then converting the `.ode` file to either `python` or `C` using `gotran2py` or `gotran2c` respectively.

## Building the `C`-demo
In order to run the `demo_c.py` you need to first compile the `C` code into a shared library. You can do this using `CMake`
```
mkdir build
cd build
cmake ..
make
```
This will create a shared library (`.dylib`, `.so` or `.dll` file) inside the `build/lib` folder. Make sure to update the line `lib = np.ctypeslib.load_library("libbeeler_reuter.dylib", "build/lib")` in `demo_c.py` to reflect your operating system. Python bindings are already implemented in `cmodel.py` using `ctypes`.
