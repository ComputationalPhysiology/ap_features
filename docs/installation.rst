.. highlight:: shell

============
Installation
============


Virtual environment
-------------------

Before you install any packages it is recommended that you create a virtual environment. You can do this using the built in `venv`_ module.
It is also possible to use the `virtualenv`_ package which can be installed with `pip`_.

.. code-block:: console

    $ python -m pip install virtualenv

Now create a virtual environment as follows:

.. code-block:: console

    $ python -m virtualenv venv

and activate the virtual environment. For unix users you can use

.. code-block:: console

    $ source venv/bin/activate

and Windows users can use

.. code-block:: console

    $ .\venv\Scripts\activate.bat


Stable release
--------------

To install Action Potential features, run this command in your terminal:

.. code-block:: console

    $ pip install ap_features

This is the preferred method to install Action Potential features, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _venv: https://docs.python.org/3/library/venv.html

From sources
------------

The sources for Action Potential features can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:finsberg/ap_features.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/finsberg/ap_features/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install .

There is also a way to install the package using the Makefile, i.e

.. code-block:: console

    $ make install

For developers
~~~~~~~~~~~~~~~

If you plan to develop this package you should also make sure to install the development dependencies listed in the `requirements_dev.txt`.
In addition you should also make sure to install the pre-commit hook. All of this can be installed by executing the command

.. code-block:: console

    $ make dev

Note that this will also install the main package in editable mode, which is nice when developing.

.. _Github repo: https://github.com/finsberg/ap_features
.. _tarball: https://github.com/finsberg/ap_features/tarball/master
