[![image](https://img.shields.io/pypi/v/ap_features.svg)](https://pypi.python.org/pypi/ap_features)
![CI](https://github.com/ComputationalPhysiology/ap_features/workflows/CI/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ComputationalPhysiology/ap_features/main.svg)](https://results.pre-commit.ci/latest/github/ComputationalPhysiology/ap_features/main)
[![github pages](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/github-pages.yml/badge.svg)](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/github-pages.yml)
[![Build and upload to PyPI](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/deploy.yml/badge.svg)](https://github.com/ComputationalPhysiology/ap_features/actions/workflows/deploy.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/ap_features-coverage.json)](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/ap_features-coverage.json)
# Action Potential features

`ap_features` is package for computing features of action potential traces. This includes chopping, background correction and feature calculations.

Parts of this library is written in `numba` and is therefore highly performant. This is useful if you want to do feature calculations on a large number of traces.

## Quick start

```python
import numpy as np
import ap_features as apf

# Time in seconds
t = np.linspace(0, 1, 100)
# Some synthetic trace of a calcium transient
y = apf.testing.ca_transient(t)
beat = apf.Beat(y=y, t=t)

print(f"APD30: {beat.apd(30):.3f}s, APD80: {beat.apd(80):.3f}s")
print(f"Time to peak: {beat.ttp():.3f}s")
print(f"Decay time from max to 90%: {beat.tau(a=0.1):.3f}s")
```

```
APD30: 0.129s, APD80: 0.306s
Time to peak: 0.121s
Decay time from max to 90%: 0.319s
```

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
