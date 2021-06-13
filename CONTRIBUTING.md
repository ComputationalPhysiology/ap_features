# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/ComputationalPhysiology/ap_features/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in
    troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to
implement it.

### Write Documentation

Action Potential features could always use more documentation,
whether as part of the official Action Potential features docs, in
docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/ComputationalPhysiology/ap_features/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to
    implement.
-   Remember that this is a volunteer-driven project, and that
    contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up ap_features for local development.

1.  Fork the ap_features repo on
    GitHub.

2.  Clone your fork locally:

    ```Shell
    $ git clone git@github.com:your_name_here/ap_features.git
    ```

3.  Install your local copy into a virtual environment.

    ```Shell
    $ cd ap_features/
    $ python -m venv venv
    $ source venv/bin/activate
    $ python -m pip install -e .
    ```

4.  Create a branch for local development:

    ```Shell
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you're done making changes, check that your changes pass
    flake8, static type checking and the tests, including testing other Python versions with
    tox:

    ```Shell
    $ make lint
    $ make type
    $ make test
    ```

    To get flake8 and tox, just pip install them into your virtualenv.

6.  Commit your changes and push your branch to GitHub:

    ```Shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.rst.
3.  The pull request should work for Python 3.7 and 3.8. Check https://github.com/ComputationalPhysiology/ap_features/actions to make sure all tests are passing.
    Python versions.

## Tips

To run a subset of tests starting with `test_something` do:

```Shell
$ python -m pytest -k test_something
```



## Deploying

A reminder for the maintainers on how to deploy. Make sure all your
changes are committed (including an entry in HISTORY.md). Then run:

```Shell
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

Travis will then deploy to PyPI if tests pass.
