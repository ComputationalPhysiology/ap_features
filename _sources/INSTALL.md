# Installation

## Virtual environment

Before you install any packages it is recommended that you create a
virtual environment. You can do this using the built in
[venv](https://docs.python.org/3/library/venv.html) module. It is also
possible to use the [virtualenv](https://virtualenv.pypa.io/en/latest/)
package which can be installed with [pip](https://pip.pypa.io).

```Shell
$ python -m pip install virtualenv
```

Now create a virtual environment as follows:

```Shell
$ python -m virtualenv venv
```

and activate the virtual environment. For unix users you can use

```Shell
$ source venv/bin/activate
```

and Windows users can use

```Shell
$ .\venv\Scripts\activate.bat
```

## Stable release

To install Action Potential features, run this command in your terminal:

```Shell
$ pip install ap_features
```

This is the preferred method to install Action Potential features, as it
will always install the most recent stable release.

If you don\'t have [pip](https://pip.pypa.io) installed, this [Python
installation
guide](http://docs.python-guide.org/en/latest/starting/installation/)
can guide you through the process.

## From sources

The sources for Action Potential features can be downloaded from the
[Github repo](https://github.com/ComputationalPhysiology/ap_features).

You can either clone the public repository:

```Shell
$ git clone git@github.com:ComputationalPhysiology/ap_features.git
```

Or download the
[tarball](https://github.com/ComputationalPhysiology/ap_features/tarball/master):

```Shell
$ curl -OJL https://github.com/ComputationalPhysiology/ap_features/tarball/master
```

Once you have a copy of the source, you can install it with:

```Shell
$ python -m pip install .
```

There is also a way to install the package using the Makefile, i.e

```Shell
$ make install
```

### For developers

If you plan to develop this package you should also make sure to install
the development dependencies using the command

```Shell
python -m pip install -e ".[dev]"
```

In addition you should also make sure to install the pre-commit hook. All of this can be installed by
executing the command

```Shell
$ make dev
```

Note that this will also install the main package in editable mode,
which is nice when developing.
