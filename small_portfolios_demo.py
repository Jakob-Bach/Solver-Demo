"""Find Small Portfolios

This file demonstrates different optimizers and problem formulations (mostly MIP but also SMT) to
tackle the K-Portfolio Problem, analyzed in the paper "A Comprehensive Study of k-Portfolios of
Recent SAT Solvers" (reference: https://doi.org/10.4230/LIPIcs.SAT.2022.2 ). This file mainly
focuses on the APIs, but does not benchmark the approaches systematically, so you won't get useful
information by running the whole file as a script. In the end, we settled on the "Python-MIP"
solution, which was fastest.
Most approaches are not very fast, so you might want to reduce the size of the input data matrix,
i.e., the number of instances (= rows) and/or solvers (= columns). Interestingly, the relationship
of runtime to parameter k (= portfolio size) is less clear. It even seems that many solvers are
slowest for some small k, but pick up speed for larger k (which actually have a bigger search space,
at least until k = n/2).
"""

import itertools

import cvxpy
import gekko
import mip
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import pandas as pd
import pulp
import z3

# Prepare dataset
NUM_INSTANCES = 30
NUM_SOLVERS = 20
np.random.seed(525)
dataset = pd.DataFrame(np.random.rand(NUM_INSTANCES, NUM_SOLVERS))
dataset.columns = [f'Solver_{i}' for i in range(dataset.shape[1])]
k = 1  # number of solvers to select
print(dataset.sum().min())  # Single Best Solver (k=1)
print(dataset.sum().idxmin())


# Exhaustive search -- fastest solution for very small k, in particular, if many instances
objective_value = float('inf')
best_solution = ()  # emtpy tuple
for solution in itertools.combinations(range(dataset.shape[1]), k):
    score = dataset.iloc[:, list(solution)].min(axis='columns').sum()
    if score < objective_value:
        objective_value = score
        best_solution = solution
print(objective_value)
print(best_solution)


# CVXPY solution 1 -- straightforward solution with only binary selection variables for solvers;
# does not work, since objective not convex (due to min function) and thus cannot be minimized
solver_vars = cvxpy.Variable(shape=dataset.shape[1], boolean=True)
instance_vbs_scores = [cvxpy.min(cvxpy.multiply(dataset.iloc[i].to_numpy(), solver_vars))
                       for i in range(len(dataset))]
problem = cvxpy.Problem(objective=cvxpy.Minimize(cvxpy.sum(instance_vbs_scores)),
                        constraints=[cvxpy.sum(solver_vars) <= k])
problem.solve()


# CVPXY solution 2 -- different formulation (using more variables) to get convexity
print(cvxpy.installed_solvers())
instance_solver_vars = cvxpy.Variable(shape=dataset.shape, boolean=True)
# Pick one solver per instance:
per_instance_constraint = cvxpy.sum(instance_solver_vars, axis=1) == 1  # a vector of constraints
# Make sure at most k solvers are used (for at least one instance each):
solver_cardinality_constraint = cvxpy.sum(cvxpy.max(instance_solver_vars, axis=0)) <= k
objective = cvxpy.Minimize(cvxpy.sum(cvxpy.multiply(instance_solver_vars, dataset.to_numpy())))
constraints = [per_instance_constraint, solver_cardinality_constraint]
problem = cvxpy.Problem(objective=objective, constraints=constraints)
print(problem.solve())
print(problem.solver_stats.solver_name)
print(instance_solver_vars.value.max(axis=0))  # solvers used for at least one instance


# Google OR-Tools solution 1: CSP
model = cp_model.CpModel()
instance_solver_vars = [[model.NewBoolVar(f'x_{i}_{j}') for j in range(dataset.shape[1])]
                        for i in range(dataset.shape[0])]
solver_vars = [model.NewBoolVar(f's_{j}')for j in range(dataset.shape[1])]
for var_list in instance_solver_vars:  # per-instance constraints
    model.Add(cp_model.LinearExpr.Sum(var_list) == 1)
for j in range(dataset.shape[1]):  # per-solver-constraints
    model.Add(cp_model.LinearExpr.Sum([instance_solver_vars[i][j] for i in range(dataset.shape[0])]) <=
              dataset.shape[0] * solver_vars[j])  # "Implies" in Z3
model.Add(cp_model.LinearExpr.Sum(solver_vars) <= k)
model.Minimize(cp_model.LinearExpr.Sum([instance_solver_vars[i][j] * float(dataset.iloc[i, j])
                                       for i in range(dataset.shape[0]) for j in range(dataset.shape[1])]))
solver = cp_model.CpSolver()
print(solver.Solve(model) == cp_model.OPTIMAL)
print(solver.ObjectiveValue())
print([solver.Value(var) for var in solver_vars])


# Google OR-Tools solution 2: LP --
# does not work, as it ignores that variables should be binary (also if IntVar instead of BoolVar)
solver = pywraplp.Solver_CreateSolver("glop")
instance_solver_vars = [[solver.BoolVar(f'x_{i}_{j}') for j in range(dataset.shape[1])]
                        for i in range(dataset.shape[0])]
solver_vars = [solver.BoolVar(f's_{j}')for j in range(dataset.shape[1])]
for var_list in instance_solver_vars:  # per-instance constraints
    solver.Add(solver.Sum(var_list) == 1)
for j in range(dataset.shape[1]):  # per-solver-constraints
    solver.Add(solver.Sum(instance_solver_vars[i][j] for i in range(dataset.shape[0])) <=
               dataset.shape[0] * solver_vars[j])  # "Implies" in Z3
solver.Add(solver.Sum(solver_vars) <= k)
solver.Minimize(solver.Sum(instance_solver_vars[i][j] * float(dataset.iloc[i, j])
                           for i in range(dataset.shape[0]) for j in range(dataset.shape[1])))
print(solver.Solve() == solver.OPTIMAL)
print(solver.Objective().Value())
print([var.solution_value() for var in solver_vars])


# Python-MIP solution
model = mip.Model()
model.verbose = 0
instance_solver_vars = [[model.add_var(f'x_{i}_{j}', var_type=mip.BINARY)
                         for j in range(dataset.shape[1])] for i in range(dataset.shape[0])]
solver_vars = [model.add_var(f's_{j}', var_type=mip.BINARY)for j in range(dataset.shape[1])]
for var_list in instance_solver_vars:  # per-instance constraints
    model.add_constr(mip.xsum(var_list) == 1)
for j in range(dataset.shape[1]):  # per-solver-constraints
    model.add_constr(mip.xsum(instance_solver_vars[i][j] for i in range(dataset.shape[0])) <=
                     dataset.shape[0] * solver_vars[j])  # "Implies" in Z3
model.add_constr(mip.xsum(solver_vars) <= k)
model.objective = mip.minimize(mip.xsum(instance_solver_vars[i][j] * float(dataset.iloc[i, j])
                                        for i in range(dataset.shape[0]) for j in range(dataset.shape[1])))
print(model.optimize())
print(model.objective_value)
print([var.x for var in solver_vars])


# PuLP solution
print(pulp.listSolvers(onlyAvailable=True))
solver = pulp.PULP_CBC_CMD(msg=False)  # default solver
problem = pulp.LpProblem(sense=pulp.const.LpMinimize)
instance_solver_vars = [[pulp.LpVariable(f'x_{i}_{j}', cat=pulp.const.LpBinary)
                         for j in range(dataset.shape[1])] for i in range(dataset.shape[0])]
solver_vars = [pulp.LpVariable(f's_{j}', cat=pulp.const.LpBinary) for j in range(dataset.shape[1])]
# Add objective first:
problem += pulp.lpSum(instance_solver_vars[i][j] * float(dataset.iloc[i, j])
                      for i in range(dataset.shape[0]) for j in range(dataset.shape[1]))
for var_list in instance_solver_vars:  # per-instance constraints
    problem += pulp.lpSum(var_list) == 1
for j in range(dataset.shape[1]):  # per-solver-constraints
    problem += pulp.lpSum(instance_solver_vars[i][j] for i in range(dataset.shape[0])) <=\
        dataset.shape[0] * solver_vars[j]  # "Implies" in Z3
problem += pulp.lpSum(solver_vars) <= k
print(problem.solve(solver=solver) == pulp.const.LpStatusOptimal)
print(problem.objective.value())
print([var.value() for var in solver_vars])


# GEKKO solution -- seems to be slower than the other integer optimizers
model = gekko.GEKKO(remote=False)
model.options.SOLVER = 1  # choose MIP solver, else solutions might not be integer
instance_solver_vars = [[model.Var(name=f'x_{i}_{j}', lb=0, ub=1, integer=True)
                         for j in range(dataset.shape[1])] for i in range(dataset.shape[0])]
solver_vars = [model.Var(name=f's_{j}', lb=0, ub=1, integer=True) for j in range(dataset.shape[1])]
for var_list in instance_solver_vars:  # per-instance constraints
    model.Equation(sum(var_list) == 1)
for j in range(dataset.shape[1]):  # per-solver-constraints
    model.Equation(sum(instance_solver_vars[i][j] for i in range(dataset.shape[0])) <=
                   dataset.shape[0] * solver_vars[j])  # "Implies" in Z3
model.Equation(sum(solver_vars) <= k)
# Objective needs to be broken down, else too big:
objective_vars = [model.Var(name=f'score_{i}') for i in range(dataset.shape[0])]
for i in range(dataset.shape[0]):
    model.Equation(sum(instance_solver_vars[i][j] * float(dataset.iloc[i, j])
                       for j in range(dataset.shape[1])) == objective_vars[i])
model.Obj(sum(objective_vars))
model.solve(disp=True)
print(model.options.objfcnval)
print([var.value[0] for var in solver_vars])


# Z3 solution -- also rather slow
z3.set_param('sat.cardinality.solver', False)  # should speed-up solving
instance_solver_vars = [[z3.Bool(f'x_{i}_{j}') for j in range(dataset.shape[1])]
                        for i in range(dataset.shape[0])]
solver_vars = [z3.Bool(f's_{j}') for j in range(dataset.shape[1])]
per_instance_constraints = [z3.And(z3.AtLeast(*var_list, 1), z3.AtMost(*var_list, 1))
                            for var_list in instance_solver_vars]
per_solver_constraints = [z3.Implies(z3.Or([instance_solver_vars[i][j] for i in range(dataset.shape[0])]),
                                     solver_vars[j]) for j in range(dataset.shape[1])]
optimizer = z3.Optimize()
objective = optimizer.minimize(z3.Sum([instance_solver_vars[i][j] * float(dataset.iloc[i, j])
                                       for i in range(dataset.shape[0]) for j in range(dataset.shape[1])]))
optimizer.add(per_instance_constraints)
optimizer.add(per_solver_constraints)
optimizer.add(z3.AtMost(*solver_vars, k))
print(optimizer.check())
print(objective.value().numerator_as_long() / objective.value().denominator_as_long())
print([int(bool(optimizer.model()[var])) for var in solver_vars])
# Serialize problem to SMT-LIB format
# with open('small_portfolio_problem.smtlib', mode='w') as outfile:
#     outfile.write(str(optimizer))

# Z3 solution 2 -- faster, using QF_LRA / QF_LIA (i.e., also use numerical variables)
# based on Nof (2020) "Real-time solving of computationally hard problems using optimal algorithm
# portfolios", but simplified (merges "value choice constrains" and "implied algorithms constraints"
# into one constraint type: for each instance, make sure there is one solver selected that has the
# runtime that goes into the objective for this instance)
z3.set_param('sat.cardinality.solver', False)  # should speed-up solving
instance_vars = [z3.Real(f'v_{i}') for i in range(dataset.shape[0])]
solver_vars = [z3.Bool(f's_{j}') for j in range(dataset.shape[1])]
value_constraints = [z3.Or([z3.And(instance_vars[i] == float(dataset.iloc[i, j]), solver_vars[j])
                            for j in range(dataset.shape[1])]) for i in range(dataset.shape[0])]
optimizer = z3.Optimize()
objective = optimizer.minimize(z3.Sum(instance_vars))
optimizer.add(value_constraints)
optimizer.add(z3.AtMost(*solver_vars, k))
print(optimizer.check())
print(objective.value().numerator_as_long() / objective.value().denominator_as_long())
print([int(bool(optimizer.model()[var])) for var in solver_vars])
