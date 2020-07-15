# Solver Demo

We demonstrate the functionality of multiple SAT/SMT solvers in multiple programming languages.
The broadest demo is for `Z3` in Python (`Z3_Demo.ipynb`).
In all other cases, we mainly benchmark how fast the solvers are in counting all solutions for a simple AND or OR formula.
(In contrast to the standard case, where one is interested in finding one solution or just knowing satisfiability.)
Nevertheless, that also demonstrates their API.

## Solvers

|Solver|Languages|Hints|
|---|---|---|
|[Choco](https://choco-solver.org/) |Java|also an optimizer|
|[Google OR Tools](https://developers.google.com/optimization/introduction/overview) |Java, Python|also an optimizer|
|[PicoSAT](https://pypi.org/project/pycosat/) |Python||
|[pySMT](https://github.com/pysmt/pysmt)|Python|wraps various solvers|
|[python-constraint](https://labix.org/python-constraint) |Python||
|[Z3](https://github.com/Z3Prover/z3/wiki) |Java, Python|also an optimizer|

For all three languages (C++, Java, Python), we also provide own solution counters/enumerators:

- tailored to the specific benchmark formulas (AND, OR), converted to an arithmetic representation
- for arbitrary logical expressions, constructed in an object-oriented manner

## Setup

### Python Code

We use `Python 3.8.3` (in particular, out of `Anaconda3 2019.10`) with [`conda 4.8.3`](https://docs.conda.io/en/latest/) dependency management.
For reproducibility purposes, we exported our environment with `conda env export --no-builds > environment.yml`.
You should be able to import the environment via `conda env create -f environment.yml`
and then activate it with `conda activate solver-demo`.
After activating, call `ipython kernel install --user --name=solver-demo` to make the environment available for notebooks.

### Java Code

We use Java 8.
For the Java code, you can import `java_solvers` as a Maven project into Eclipse.

`ArithmeticEnumerationDemo` and `FlexibleEnumerationDemo` don't have any external dependencies.

`ChocoDemo` should also work directly, its dependency is hosted on Maven Central.

For `Z3Demo`, matters are more complicated.
First, please download a [pre-built version of Z3](https://github.com/Z3Prover/z3/releases) and extract it.
(If that's too easy for you, you can also try to compile it.)
Our project also has a Maven dependency on `Z3`, but the Z3 download only provides a plain JAR.
Thus, extract [Maven](https://maven.apache.org/download.cgi) somewhere on your computer and add it to your `PATH` (optional).
Next, install the Z3 JAR into your local Maven repository (might need to adapt file path and version):

```
mvn install:install-file -Dfile=com.microsoft.z3.jar -DgroupId=com.microsoft -DartifactId=z3 -Dversion=4.8.7 -Dpackaging=jar
```

Furthermore, the JAR depends on DLLs also included in the Z3 download.
To enable access, add the `bin/` directory of your Z3 download to the environment variable `Path`.

For `ORToolsDemo`, the process is similar, download is [here](https://developers.google.com/optimization/install/download) , but you need to build two Maven artifacts.
