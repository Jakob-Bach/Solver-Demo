"""Benchmark filter feature selection with OR-Tools (MIP formulation) and Z3 (SMT formulation)

Implements various filter-feature-selection methods and benchmarks them.

Usage: python -m filter_fs_z3_benchmark --help
"""


import argparse
import multiprocessing
import time
from typing import Any, Callable, Dict, List, Tuple

from ortools.linear_solver import pywraplp
import pandas as pd
import sklearn.datasets
import tqdm
import z3


# Prepare dataset (store it in global scope)
dataset = sklearn.datasets.fetch_california_housing()  # alternative in sklearn: load_diabetes()
X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name=dataset.target_names[0])
target_correlation = X.corrwith(y).abs().values
feature_correlation = X.corr().abs().values


# Run OR-Tools and return a tuple (objective value, list of binary selection decisions)
def optimize_mip(optimizer, selection_variables) -> Tuple[float, List[bool]]:
    optimizer.Solve()
    objective_value = optimizer.Objective().Value()
    selection = [bool(var.solution_value()) for var in selection_variables]
    return objective_value, selection


# Run Z3 and return a tuple (objective value, list of binary selection decisions)
def optimize_smt(optimizer: z3.Optimize, objective: z3.z3.OptimizeObjective,
                 selection_variables: List[z3.z3.BoolRef]) -> Tuple[float, List[bool]]:
    optimizer.check()
    if isinstance(objective.value(), z3.IntNumRef):
        objective_value = objective.value()
    else:  # RatNumRef
        objective_value = (objective.value().numerator_as_long() /
                           objective.value().denominator_as_long())
    selection = [str(optimizer.model()[var]) == 'True' for var in selection_variables]
    return objective_value, selection


# Univariate feature scoring, without redundancy terms
# "k" = number of features to be selected (else all features selected, as unconstrained problem)
def univariate_optimizer_mip(k: int = 3) -> Tuple[float, List[bool]]:
    optimizer = pywraplp.Solver_CreateSolver('CBC')
    selection_variables = [optimizer.BoolVar('x_' + str(i)) for i in range(X.shape[1])]
    objective = optimizer.Sum([var * val for (var, val) in zip(selection_variables, target_correlation)])
    optimizer.Maximize(objective)
    optimizer.Add(optimizer.Sum(selection_variables) <= k)
    return optimize_mip(optimizer=optimizer, selection_variables=selection_variables)


def univariate_optimizer_smt(k: int = 3) -> Tuple[float, List[bool]]:
    optimizer = z3.Optimize()
    selection_variables = z3.Bools(' '.join(['x_' + str(i) for i in range(X.shape[1])]))
    objective = z3.Sum(*[z3.If(var, val, 0) for var, val in zip(selection_variables, target_correlation)])
    objective = optimizer.maximize(objective)
    optimizer.add(z3.AtMost(*selection_variables, k))
    return optimize_smt(optimizer=optimizer, objective=objective,
                        selection_variables=selection_variables)


# CFS (Correlation-based Feature Selection) -- Hall et al. (1999): "Feature Selection for Machine
# Learning: Comparing a Correlation-based Filter Approach to the Wrapper"
# also, see https://en.wikipedia.org/wiki/Feature_selection#Correlation_feature_selection
def cfs_optimizer_smt() -> Tuple[float, List[bool]]:
    optimizer = z3.Optimize()
    selection_variables = z3.Bools(' '.join(['x_' + str(i) for i in range(X.shape[1])]))
    relevance = z3.Sum(*[z3.If(var, val, 0) for var, val in zip(selection_variables, target_correlation)])
    relevance = relevance * relevance  # we optimize squared "Merit" (so there is no square root in denominator)
    redundancy = z3.Sum(*[z3.If(z3.And(selection_variables[i], selection_variables[j]),
                                feature_correlation[i, j], 0)
                          for i in range(len(selection_variables))
                          for j in range(len(selection_variables))
                          if i != j])
    k = z3.Sum(*[z3.If(var, 1, 0) for var in selection_variables])
    redundancy = k + redundancy
    objective = relevance / redundancy
    objective = optimizer.maximize(objective)
    optimizer.add(z3.Or(selection_variables))  # one feature needs to be selected, else div by zero
    return optimize_smt(optimizer=optimizer, objective=objective,
                        selection_variables=selection_variables)


# FCBF (Fast Correlation-Based Filter) -- Yu et. al (2003): "Feature Selection for High-Dimensional
# Data: A Fast Correlation-Based Filter Solution"
# Rephrased as optimization problem with an objective, while the original paper only has constraints
# and searches for valid solutions heuristically.
# "k" = number of features to be selected (procedure would also work without specifying this)
# "delta" = minimum relevance of selected features (even if 0, number of features reduced because
# redundancy constraints)
def fcbf_optimizer_mip(k: int = 3, delta: float = 0) -> Tuple[float, List[bool]]:
    optimizer = pywraplp.Solver_CreateSolver('CBC')
    selection_variables = [optimizer.BoolVar('x_' + str(i)) for i in range(X.shape[1])]
    objective = optimizer.Sum([var * val for (var, val) in zip(selection_variables, target_correlation)])
    optimizer.Maximize(objective)
    optimizer.Add(optimizer.Sum(selection_variables) <= k)
    for i in range(len(selection_variables)):  # select only features which are predominant
        # Condition 1 (relevance) for predominance: dependency to target has to be over threshold
        if target_correlation[i] < delta:
            optimizer.Add(selection_variables[i] == 0)
        # Condition 2 (redundancy) for predominance: no other feature selected that has higher
        # dependency to current feature than current feature has to target
        for j in range(i):
            if ((target_correlation[i] <= feature_correlation[j, i]) or
                (target_correlation[j] <= feature_correlation[i, j])):
                optimizer.Add(selection_variables[i] + selection_variables[j] <= 1)
    return optimize_mip(optimizer=optimizer, selection_variables=selection_variables)


def fcbf_optimizer_smt(k: int = 3, delta: float = 0) -> Tuple[float, List[bool]]:
    optimizer = z3.Optimize()
    selection_variables = z3.Bools(' '.join(['x_' + str(i) for i in range(X.shape[1])]))
    objective = z3.Sum(*[z3.If(var, val, 0) for var, val in zip(selection_variables, target_correlation)])
    objective = optimizer.maximize(objective)
    optimizer.add(z3.AtMost(*selection_variables, k))
    for i in range(len(selection_variables)):  # select only features which are predominant
        # Condition 1 (relevance) for predominance: dependency to target has to be over threshold
        optimizer.add(z3.Implies(selection_variables[i], z3.BoolVal(target_correlation[i] >= delta)))
        # Condition 2 (redundancy) for predominance: no other feature selected that has higher
        # dependency to current feature than current feature has to target
        optimizer.add(z3.And([z3.Implies(z3.And(selection_variables[i], selection_variables[j]),
                                         z3.Not(z3.BoolVal(feature_correlation[j, i] >= target_correlation[i])))
                              for j in range(len(selection_variables)) if i != j]))
    return optimize_smt(optimizer=optimizer, objective=objective,
                        selection_variables=selection_variables)


# mRMR (Minimal Redundancy Maximal Relevance) -- Peng et al. (2005): "Feature Selection Based on
# Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy"
# also, see https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_(mRMR)_feature_selection
def mrmr_optimizer_smt() -> Tuple[float, List[bool]]:
    optimizer = z3.Optimize()
    selection_variables = z3.Bools(' '.join(['x_' + str(i) for i in range(X.shape[1])]))
    k = z3.Sum(*[z3.If(var, 1, 0) for var in selection_variables])
    relevance = z3.Sum(*[z3.If(var, val, 0) for var, val in zip(selection_variables, target_correlation)])
    relevance = relevance / k
    redundancy = z3.Sum(*[z3.If(z3.And(selection_variables[i], selection_variables[j]),
                                feature_correlation[i, j], 0)
                          for i in range(len(selection_variables))
                          for j in range(len(selection_variables))])
    redundancy = redundancy / (k * k)
    objective = relevance - redundancy
    objective = optimizer.maximize(objective)
    optimizer.add(z3.Or(selection_variables))  # one feature needs to be selected, else div by zero
    return optimize_smt(optimizer=optimizer, objective=objective,
                        selection_variables=selection_variables)


# Functions used in the benchmark:
FS_FUNCTIONS = [univariate_optimizer_mip, univariate_optimizer_smt,
                cfs_optimizer_smt, fcbf_optimizer_mip, fcbf_optimizer_smt, mrmr_optimizer_smt]


# Run one feature-selection function once and return results as dictionary.
def run_one_benchmark(func: Callable[[], Tuple[float, List[bool]]]) -> Dict[str, Any]:
    start_time = time.process_time()
    objective, selection = func()
    end_time = time.process_time()
    return {'method': func.__name__.replace('_optimizer', ''), 'time': end_time - start_time,
            'objective': objective, 'selection': selection}


# Main method, running multiple FS methods for "n_iterations" each, parallelizing runs over
# "n_processes". Return DataFrame with results from all runs.
def run_benchmarks(n_processes: int, n_iterations: int) -> pd.DataFrame:
    process_pool = multiprocessing.Pool(processes=n_processes)
    progress_bar = tqdm.tqdm(total=len(FS_FUNCTIONS) * n_iterations)
    async_results = [process_pool.apply_async(run_one_benchmark, kwds={
        'func': fs_function}, callback=lambda x: progress_bar.update())
        for _ in range(n_iterations) for fs_function in FS_FUNCTIONS]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    return pd.DataFrame([x.get() for x in async_results])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarks filter-feature-selection methods',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    parser.add_argument('-i', '--iterations', type=int, default=10, dest='n_iterations',
                        help='Number of repetitions for each feature-selection method.')
    results = run_benchmarks(**vars(parser.parse_args()))
    results['selection'] = results['selection'].apply(tuple)
    print('---Runtime (in s)---')
    print(results.groupby('method')['time'].describe().round(2))
    print('---Number of different objective values---')
    print(results.groupby('method')['objective'].nunique())
    print('---Number of different feature sets---')
    print(results.groupby('method')['selection'].nunique())
