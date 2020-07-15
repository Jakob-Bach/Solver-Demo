# Solver Demo

## Solvers

|Solver|Languages|Hints|
|---|---|---|
|[Google OR Tools](https://developers.google.com/optimization/introduction/overview) |Python|also an optimizer|
|[PicoSAT](https://pypi.org/project/pycosat/) |Python||
|[pySMT](https://github.com/pysmt/pysmt)|Python|wraps various solvers|
|[python-constraint](https://labix.org/python-constraint) |Python||
|[Z3](https://github.com/Z3Prover/z3/wiki) |Python|also an optimizer|

## Setup

### Python Code

We use `Python 3.8.3` (in particular, out of `Anaconda3 2019.10`) with [`conda 4.8.3`](https://docs.conda.io/en/latest/) dependency management.
For reproducibility purposes, we exported our environment with `conda env export --no-builds > environment.yml`.
You should be able to import the environment via `conda env create -f environment.yml`
and then activate it with `conda activate solver-demo`.
After activating, call `ipython kernel install --user --name=solver-demo` to make the environment available for notebooks.
