"""Find Functional Dependencies

Search for functional dependencies in the sense of database theory, i.e., sets of left-hand-side
(LHS) attributes (= columns) in the relation (= table) that uniquely identify the values of the
right-hand-side (RHS) attributes (= columns) for each tuple (= row).

One exhaustive (manual) search procedure and one SMT-solver-based search procedure.
"""


import itertools
import time
from typing import Optional, Sequence, Tuple

import pandas as pd
import sklearn.datasets
import z3


# Check if the "dataset" contains the functional dependency "lhs_cols" -> "rhs_cols", i.e.,
# if each value combination of the LHS columns is only associated with one value combination of the
# RHS columns.
def is_fd(dataset: pd.DataFrame, lhs_cols: Sequence[str], rhs_cols: Sequence[str]) -> bool:
    return (dataset.groupby(lhs_cols).apply(lambda x: x.groupby(rhs_cols).ngroups) == 1).all()


# Find all functional dependencies in the "dataset", considering sets of up to "max_lhs_size"
# columns for the LHS and sets of up to "max_rhs_size" columns for the RHS. If no value is set for
# these parameters, then up to all columns may be used. If "allow_overlap", LHS and RHS may share
# columns (which causes the search to take longer), else not. Returns a list of tuples (LHS, RHS),
# where LHS and LHS are list of column names.
def search_fds_manually(dataset: pd.DataFrame, max_lhs_size: Optional[int] = None,
                        max_rhs_size: Optional[int] = None, allow_overlap: bool = True) \
        -> Sequence[Tuple[Sequence[str], Sequence[str]]]:
    if max_lhs_size is None:
        max_lhs_size = len(dataset.columns)
    if max_rhs_size is None:
        max_rhs_size = len(dataset.columns)
    results = []
    for lhs_size in range(1, max_lhs_size + 1):
        for lhs_cols in itertools.combinations(dataset.columns, lhs_size):
            lhs_cols = list(lhs_cols)
            if allow_overlap:
                current_max_rhs_size = max_rhs_size
            else:
                current_max_rhs_size = min(max_rhs_size, len(dataset.columns) - lhs_size)
            for rhs_size in range(1, current_max_rhs_size + 1):
                if allow_overlap:
                    current_rhs_col_candidates = dataset.columns
                else:
                    current_rhs_col_candidates = set(dataset.columns) - set(lhs_cols)
                for rhs_cols in itertools.combinations(current_rhs_col_candidates, rhs_size):
                    rhs_cols = list(rhs_cols)
                    if is_fd(dataset=dataset, lhs_cols=lhs_cols, rhs_cols=rhs_cols):
                        results.append((lhs_cols, rhs_cols))
    return results


# Find all functional dependencies with an SMT solver. Same parametrization as the manual search.
# Generalization of an encoding presented in the lecture "Parametrisierte Algorithmen" (WS 22/23),
# Chapter 13: https://scale.iti.kit.edu/_media/teaching/2022ws/param_algo/13-w3-relationale-db-print.pdf
# In particular, we additionally allow LHS and RHS to overlap (if desired by user) and we allow
# RHSs with more than one column.
def search_fds_solver(dataset: pd.DataFrame, max_lhs_size: Optional[int] = None,
                      max_rhs_size: Optional[int] = None, allow_overlap: bool = True) \
        -> Sequence[Tuple[Sequence[str], Sequence[str]]]:
    if max_lhs_size is None:
        max_lhs_size = len(dataset.columns)
    if max_rhs_size is None:
        max_rhs_size = len(dataset.columns)
    results = []
    solver = z3.Optimize()  # We don't optimize here, but z3.Solver left some variables unassigned
    # Binary decision variables: Each column can be in LHS (or not) and RHS (or not)
    lhs_vars = [z3.Bool(f'lhs_{i}') for i in range(len(dataset.columns))]
    rhs_vars = [z3.Bool(f'rhs_{i}') for i in range(len(dataset.columns))]
    # Constraint type 1: Bound number of columns in LHS and RHS
    solver.add(z3.AtLeast(*lhs_vars, 1))
    solver.add(z3.AtMost(*lhs_vars, max_lhs_size))
    solver.add(z3.AtLeast(*rhs_vars, 1))
    solver.add(z3.AtMost(*rhs_vars, max_rhs_size))
    # Constraint type 2: If desired, prevent columns appearing on both sides
    if not allow_overlap:
        for (lhs_var, rhs_var) in zip(lhs_vars, rhs_vars):
            solver.add(z3.Not(z3.And(lhs_var, rhs_var)))
    # Constraint type 3: Make sure LHS identifies RHS uniquely -> For each pair of tuples and each
    # RHS column (fortunately, we need not iterate over colum combinations), if the tuples differ
    # in their RHS column, then they should also differ in at least one value of their LHS columns
    for i in range(0, len(dataset) - 1):  # First tuple
        for j in range(i, len(dataset)):  # Second tuple
            is_col_value_diff = dataset.iloc[i] != dataset.iloc[j]
            lhs_diff_vars = [var for var, is_diff in zip(lhs_vars, is_col_value_diff) if is_diff]
            rhs_diff_vars = [var for var, is_diff in zip(rhs_vars, is_col_value_diff) if is_diff]
            solver.add(z3.Implies(z3.Or(rhs_diff_vars), z3.Or(lhs_diff_vars)))
    while solver.check() == z3.sat:  # Enumerate all valid solutions
        results.append(([col for col, lhs_var in zip(dataset.columns, lhs_vars)
                         if bool(solver.model()[lhs_var])],
                        [col for col, rhs_var in zip(dataset.columns, rhs_vars)
                         if bool(solver.model()[rhs_var])]))
        # Prevent finding previous solution again by requiring at least one variable to take a
        # different value:
        solver.add(z3.Or(z3.Or([z3.Not(lhs_var) if bool(solver.model()[lhs_var]) else lhs_var
                               for lhs_var in lhs_vars]),
                         z3.Or([z3.Not(rhs_var) if bool(solver.model()[rhs_var]) else rhs_var
                                for rhs_var in rhs_vars])))
    return results


# Global parameters, may be modified by the user.
MAX_LHS_SIZE = None  # None or an integer; controls max number of columns in LHS of dependency
MAX_RHS_SIZE = 1  # None or an integer; controls max number of columns in RHS of dependency
ALLOW_OVERLAP = False  # boolean; controls whether LHS and RHS of dependency may share columns


if __name__ == '__main__':
    X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    demo_dataset = pd.concat([X, y], axis='columns')
    print(f'Demo dataset has {demo_dataset.shape[0]} rows and {demo_dataset.shape[1]} columns.')

    start_time = time.perf_counter()
    result = search_fds_manually(dataset=demo_dataset, max_lhs_size=MAX_LHS_SIZE,
                                 max_rhs_size=MAX_RHS_SIZE, allow_overlap=ALLOW_OVERLAP)
    end_time = time.perf_counter()
    print('[Manual search] Number of functional dependencies:', len(result))
    print('[Manual search] Time:', round(end_time - start_time, 3), 's')

    start_time = time.perf_counter()
    result = search_fds_solver(dataset=demo_dataset, max_lhs_size=MAX_LHS_SIZE,
                               max_rhs_size=MAX_RHS_SIZE, allow_overlap=ALLOW_OVERLAP)
    end_time = time.perf_counter()
    print('[Solver search] Number of functional dependencies:', len(result))
    print('[Solver search] Time:', round(end_time - start_time, 3), 's')
