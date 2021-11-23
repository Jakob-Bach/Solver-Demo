# Solver Demo

We demonstrate and benchmark the functionality of multiple SAT/SMT solvers/optimizers in multiple programming languages.
We have two use cases:

- use case 1: counting (by enumerating) all solutions for a simple AND or OR formula
- use case 2: solving a simple knapsack problem

This project serves as playground for our research project on [constrained feature selection](https://github.com/Jakob-Bach/CFFS).

## Solvers

|Solver|Languages|Hints|
|---|---|---|
|[Choco](https://choco-solver.org/) |Java|also an optimizer|
|[Gecode](https://www.gecode.org/) | C++|also an optimizer|
|[GEKKO](https://gekko.readthedocs.io/en/latest/)|Python|mainly an optimizer|
|[Google OR Tools](https://developers.google.com/optimization/introduction/overview) |Java, Python|also an optimizer|
|[PicoSAT](https://pypi.org/project/pycosat/) |Python||
|[pySMT](https://github.com/pysmt/pysmt)|Python|wraps various solvers (e.g., `MathSAT`, `Z3`)|
|[python-constraint](https://labix.org/python-constraint) |Python||
|[Z3](https://github.com/Z3Prover/z3/wiki) |C++, Java, Python|also an optimizer|

For all three languages (C++, Java, Python), we also implement own solution counters/enumerators (regarding use case 1):

- option 1 ("arithmetic enumeration"): tailored to the specific benchmark formulas (AND, OR), converted to an arithmetic representation
- option 2 ("flexible enumeration"): for arbitrary logical expressions, constructed in an object-oriented manner

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

`ChocoDemo` and `ChocoOptimizationDemo` should also work directly, as their dependency is hosted on Maven Central.

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

### C++ Code

`arithmetic_enumeration_demo` and both `flexible_enumeration_demo`s don't have any external dependencies.

For `z3_demo`, you need the pre-built version of `Z3`, as for the Java pendant.
You can put the C++ file in a Visual Studio project.
Make sure to adapt (for all configurations, all platforms):

- `Project -> Properties -> Configuration Properties -> C/C++ -> General -> Additional Include Directories` by adding the path to the `include/` directory of the Z3 download.
- `Project -> Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories` by adding the path to the `bin/` directory of the Z3 download.
- `Project -> Properties -> Configuration Properties -> Linker -> Input -> Additional Dependencies` by adding `libz3.lib`.

For `gecode_demo`, you also need to [install](https://www.gecode.org/download.html) or compile the software and reference the DLLs in a similar manner.
See the [documentation](https://www.gecode.org/doc-latest/MPG.pdf) for more details.
