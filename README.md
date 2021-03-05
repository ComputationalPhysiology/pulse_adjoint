
[![CircleCI](https://circleci.com/gh/ComputationalPhysiology/pulse_adjoint.svg?style=shield)](https://circleci.com/gh/ComputationalPhysiology/pulse_adjoint)
[![Documentation Status](https://readthedocs.org/projects/pulse_adjoint/badge/?version=latest)](https://pulse-adjoint.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ComputationalPhysiology/pulse_adjoint/branch/master/graph/badge.svg?token=PG2JS1SPKJ)](https://codecov.io/gh/ComputationalPhysiology/pulse_adjoint)


# Pulse-Adjoint

Pulse-Adjoint is an extension of the [pulse](https://github.com/ComputationalPhysiology/pulse) library for doing data assimilation for cardiac mechanics applications.


## Note
This is a full rewriting of the code in the [following repo](https://github.com/ComputationalPhysiology/pulse_adjoint). Note that the version in this repo is experimental an might contains several bugs. If you encounter bugs we would appreciate if you could file an issue.


## Installation
Before installing pulse-adjoint you need to [install FEniCS](https://fenicsproject.org/download/)
See https://pulse-adjoint.readthedocs.io/en/latest/installation.html

## Documentation
Documentation can be found at https://pulse-adjoint.readthedocs.io/
You can create documentation yourselves by typing `make docs` in the
root directory.

## Getting started
Check out the demos in the demo folder.

## Automated test
Test are provided in the folder [`tests`](tests). You can run the test
with `pytest`
```
python -m pytest tests -vv
```
You can also type
```
make test
```
which with run the test and generate a coverage report. You can have a look at the coverage report if you type
```
make coverage
```

Tests are run in continuous integration using [circle ci](https://circleci.com/gh/ComputationalPhysiology/pulse_adjoint) and a [coverage report](https://codecov.io/gh/ComputationalPhysiology/pulse_adjoint) is automatically generated


## Licence
LGPL version 3 or later


## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[finsberg/cookiecutter-pypackage](https://github.com/finsberg/cookiecutter-pypackage)
project template.
