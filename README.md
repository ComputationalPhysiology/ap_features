[![image](https://img.shields.io/pypi/v/ap_features.svg)](https://pypi.python.org/pypi/ap_features)
![CI](https://github.com/ComputationalPhysiology/ap_features/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/ComputationalPhysiology/ap_features/branch/master/graph/badge.svg?token=H0U23J41OQ)](https://codecov.io/gh/ComputationalPhysiology/ap_features)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ComputationalPhysiology/ap_features/master.svg)](https://results.pre-commit.ci/latest/github/ComputationalPhysiology/ap_features/master)
[![github pages](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/github-pages.yml/badge.svg)](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/github-pages.yml)
[![Build and upload to PyPI](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/deploy.yml/badge.svg)](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/deploy.yml)

# Action Potential features

`ap_features` is package for computing features of action potential traces. This includes chopping, background correction and feature calculations.

Parts of this library is written in `C` and `numba` and is therefore highly performant. This is useful if you want to do feature calculations on a large number of traces.

## Install
Install the package with pip
```
python -m pip install ap_features
```
See [installation instructions](https://computationalphysiology.github.io/ap_features/INSTALL.html) for more options.


## Available features
The list of currently implemented features are as follows
- Action potential duration (APD)
- Corrected action potential duration (cAPD)
- Decay time (Time for the signal amplitude to go from maximum to (1 - a) * 100 % of maximum)
- Time to peak (ttp)
- Upstroke time (time from (1-a)*100 % signal amplitude to peak)
- Beating frequency
- APD up (The duration between first intersections of two APD lines)
- Maximum relative upstroke velocity
- Maximum upstroke velocity
- APD integral (integral of the signals above the APD line)


## Documentation
Documentation is hosted at GitHub pages: <https://computationalphysiology.github.io/ap_features/>

Note that the documentation is written using [`jupyterbook`](https://jupyterbook.org) and contains an [interactive demo](https://computationalphysiology.github.io/ap_features/demo_fitzhugh_nagumo.html)


## License
* Free software: LGPLv2.1

## Source Code
* <https://github.com/ComputationalPhysiology/ap_features>
