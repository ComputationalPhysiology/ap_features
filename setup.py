#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = []

setup(
    author="Henrik Finsberg",
    author_email='henriknf@simula.no',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Package to compute features of traces from action potential models",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='ap_features',
    name='ap_features',
    packages=find_packages(include=['ap_features', 'ap_features.*']),
    test_suite='tests',
    url='https://github.com/finsberg/ap_features',
    version='0.1.0',
    project_urls={
        "Documentation": "https://ap-features.readthedocs.io.",
        "Source": "https://github.com/finsberg/ap_features",
    },
    zip_safe=False,
)
