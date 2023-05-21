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
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import ap_features as apf

time = np.linspace(0, 999, 1000)
res = solve_ivp(
    apf.testing.fitzhugh_nagumo,
    [0, 1000],
    [0.0, 0.0],
    t_eval=time,
)
trace = apf.Beats(y=res.y[0, :], t=time)
print(f"Number of beats: {trace.num_beats}")
print(f"Beat rates: {trace.beat_rates}")

# Get a list of beats
beats = trace.beats
# Pick out the second beat
beat = beats[1]

# Compute features
print(f"APD30: {beat.apd(30):.3f}s, APD80: {beat.apd(80):.3f}s")
print(f"cAPD30: {beat.capd(30):.3f}s, cAPD80: {beat.capd(80):.3f}s")
print(f"Time to peak: {beat.ttp():.3f}s")
print(f"Decay time from max to 90%: {beat.tau(a=0.1):.3f}s")
```

```
Number of beats: 5
Beat rates: [779.2207792207793, 769.2307692307693, 779.2207792207793, 759.493670886076]
APD30: 37.823s, APD80: 56.564s
cAPD30: 88.525s, cAPD80: 132.387s
Time to peak: 21.000s
Decay time from max to 90%: 53.618s
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
