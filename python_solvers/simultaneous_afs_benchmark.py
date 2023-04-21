"""Benchmark Solvers for Simultaneous Alternative Feature Selection


Try different MIP solvers (and one SMT solver) for the task of finding sufficiently different
high-quality feature sets simultaneously (i.e., find all of them at once rather than one after each
other). We assume univariates filter-feature selection, i.e.,
1) the objective function simply is the sum of the individual qualities of selected features
2) there are no constraints except limiting the size of feature sets and making sure that feature
sets are alternative enough.
"""

import random
import time


n = 50  # total number of features in the dataset
k = 5  # number of features to be selected (per feature set)
num_alternatives = 4  # number of feature sets - 1 (there also will be an "original" feature set)
tau_abs = 2  # each pair of feature sets needs to differ in at least that many features

random.seed(25)
qualities = [random.uniform(0, 1) for _ in range(n)]  # simulated quality of each feature

# ----- Python-MIP -----
import mip

model = mip.Model(sense=mip.MAXIMIZE)
model.verbose = 0
s_list = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [model.add_var(name=f's{i}_{j}', var_type=mip.BINARY) for j in range(n)]
    model.add_constr(mip.xsum(s) == k)
    for s2 in s_list:
        product_vars = []
        for (var1, var2) in zip(s, s2):
            var_name = var1.name + '*' + var2.name
            var = model.add_var(name=var_name, var_type=mip.BINARY)
            model.add_constr(var <= var1)
            model.add_constr(var <= var2)
            model.add_constr(1 + var >= var1 + var2)
            product_vars.append(var)
        model.add_constr(mip.xsum(product_vars) <= k - tau_abs)
    s_list.append(s)
objective = mip.xsum(mip.xsum(q_j * s_j for (q_j, s_j) in zip(qualities, s)) for s in s_list)
model.objective = objective
start_time = time.perf_counter()
model.optimize()
end_time = time.perf_counter()
print('MIP objective:', round(model.objective_value, 2))  # or "objective.x" or "model.objective.x"
print('MIP time:', round(end_time - start_time, 2))
mip_selected = [[j for j, s_j in enumerate(s) if s_j.x] for s in s_list]


# ----- PuLP -----
import pulp

solver = pulp.PULP_CBC_CMD(msg=False)
problem = pulp.LpProblem(sense=pulp.const.LpMaximize)
s_list = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [pulp.LpVariable(name=f's{i}_{j}', cat=pulp.const.LpBinary) for j in range(n)]
    problem += pulp.lpSum(s) == k
    for s2 in s_list:
        product_vars = []
        for (var1, var2) in zip(s, s2):
            var_name = var1.name + '*' + var2.name
            var = pulp.LpVariable(name=var_name, cat=pulp.const.LpBinary)
            problem += var <= var1
            problem += var <= var2
            problem += 1 + var >= var1 + var2
            product_vars.append(var)
        problem += pulp.lpSum(product_vars) <= k - tau_abs
    s_list.append(s)
objective = pulp.lpSum(pulp.lpSum(q_j * s_j for (q_j, s_j) in zip(qualities, s)) for s in s_list)
problem += objective
start_time = time.perf_counter()
problem.solve(solver=solver)
end_time = time.perf_counter()
print('PuLP objective:', round(problem.objective.value(), 2))  # or "objective.value()"
print('PuLP time:', round(end_time - start_time, 2))
pulp_selected = [[j for j, s_j in enumerate(s) if s_j.value()] for s in s_list]


# ----- Z3 -----
# slowest solver (comment this out or decrease problem size) and runtime very unpredictable
import z3

z3.set_param('sat.cardinality.solver', False)
optimizer = z3.Optimize()
s_list = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [z3.Bool(name=f's{i}_{j}') for j in range(n)]
    optimizer.add(z3.Sum([z3.If(s_j, 1, 0) for s_j in s]) == k)
    for s2 in s_list:
        optimizer.add(z3.Sum([z3.If(z3.And(s_1_j, s_2_j), 1, 0)
                              for (s_1_j, s_2_j) in zip(s, s2)]) == k - tau_abs)
    s_list.append(s)
objective = optimizer.maximize(z3.Sum([z3.If(s_j, q_j, 0) for s in s_list
                                        for (q_j, s_j) in zip(qualities, s)]))
start_time = time.perf_counter()
optimizer.check()
end_time = time.perf_counter()
print('Z3 objective:', objective.value())
print('Z3 time:', round(end_time - start_time, 2))
z3_selected = [[j for j, s_j in enumerate(s) if bool(optimizer.model()[s_j])] for s in s_list]


# ----- CVXPY -----
import cvxpy

s_list = []
constraints = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [cvxpy.Variable(name=f's{i}_{j}', boolean=True) for j in range(n)]
    constraints.append(cvxpy.sum(s) == k)
    for s2 in s_list:
        product_vars = []
        for (var1, var2) in zip(s, s2):
            var_name = var1.name() + '*' + var2.name()
            var = cvxpy.Variable(name=var_name, boolean=True)
            constraints.append(var <= var1)
            constraints.append(var <= var2)
            constraints.append(1 + var >= var1 + var2)
            product_vars.append(var)
        constraints.append(cvxpy.sum(product_vars) <= k - tau_abs)
        # constraints.append(cvxpy.scalar_product(s, s2) <= k - tau_abs)  # violates DCP rules
    s_list.append(s)
objective = cvxpy.Maximize(cvxpy.sum([cvxpy.sum([q_j * s_j for (q_j, s_j) in zip(qualities, s)])
                                      for s in s_list]))
problem = cvxpy.Problem(objective=objective, constraints=constraints)
start_time = time.perf_counter()
problem.solve()
end_time = time.perf_counter()
print('CVXPY objective:', round(objective.value, 2))
print('CVXPY time:', round(end_time - start_time, 2))
cvxpy_selected = [[j for j, s_j in enumerate(s) if s_j.value] for s in s_list]


# ----- Google OR Tools: CSP -----
from ortools.sat.python import cp_model

model = cp_model.CpModel()
s_list = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [model.NewBoolVar(name=f's{i}_{j}') for j in range(n)]
    model.Add(cp_model.LinearExpr.Sum(s) == k)
    for s2 in s_list:
        product_vars = []
        for (var1, var2) in zip(s, s2):
            var_name = var1.Name() + '*' + var2.Name()
            var = model.NewBoolVar(name=var_name)
            model.Add(var <= var1)
            model.Add(var <= var2)
            model.Add(1 + var >= var1 + var2)
            product_vars.append(var)
        model.Add(cp_model.LinearExpr.Sum(product_vars) <= k - tau_abs)
    s_list.append(s)
objective = cp_model.LinearExpr.Sum([cp_model.LinearExpr.WeightedSum(s, qualities) for s in s_list])
model.Maximize(objective)
solver = cp_model.CpSolver()
start_time = time.perf_counter()
solver.Solve(model)
end_time = time.perf_counter()
print('OR-Tools (CSP) objective:', round(solver.ObjectiveValue(), 2))
print('OR-Tools (CSP) time:', round(end_time - start_time, 2))
ortools_csp_selected = [[j for j, s_j in enumerate(s) if solver.Value(s_j)] for s in s_list]


# ----- Google OR Tools: LP -----
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver_CreateSolver('CBC')  # other solvers possible, see doc of function
s_list = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [solver.BoolVar(name=f's{i}_{j}') for j in range(n)]
    solver.Add(solver.Sum(s) == k)
    for s2 in s_list:
        product_vars = []
        for (var1, var2) in zip(s, s2):
            var_name = var1.name() + '*' + var2.name()
            var = solver.BoolVar(name=var_name)
            solver.Add(var <= var1)
            solver.Add(var <= var2)
            solver.Add(1 + var >= var1 + var2)
            product_vars.append(var)
        solver.Add(solver.Sum(product_vars) <= k - tau_abs)
    s_list.append(s)
objective = solver.Sum([solver.Sum([q_j * s_j for (q_j, s_j) in zip(qualities, s)]) for s in s_list])
solver.Maximize(objective)
start_time = time.perf_counter()
solver.Solve()
end_time = time.perf_counter()
print('OR-Tools (LP; standard objective) objective:', round(solver.Objective().Value(), 2))
print('OR-Tools (LP; standard objective) individual objectives:',
      [round(sum(q_j * s_j.solution_value() for (q_j, s_j) in zip(qualities, s)), 2) for s in s_list])
print('OR-Tools (LP; standard objective) time:', round(end_time - start_time, 2))
ortools_lp_selected = [[j for j, s_j in enumerate(s) if s_j.solution_value()] for s in s_list]

# ----- Google OR Tools: LP, min-quality objective -----
# instead of maximizing summed quality over all feature sets, maximize minimum quality of all
# feature sets (to get more even distribution of feature-set quality)
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver_CreateSolver('CBC')  # other solvers possible, see doc of function
s_list = []
Q_list = []
Q_min = solver.NumVar(name='Q_min', lb=0, ub=sum(qualities))
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [solver.BoolVar(name=f's{i}_{j}') for j in range(n)]
    solver.Add(solver.Sum(s) == k)
    for s2 in s_list:
        product_vars = []
        for (var1, var2) in zip(s, s2):
            var_name = var1.name() + '*' + var2.name()
            var = solver.BoolVar(name=var_name)
            solver.Add(var <= var1)
            solver.Add(var <= var2)
            solver.Add(1 + var >= var1 + var2)
            product_vars.append(var)
        solver.Add(solver.Sum(product_vars) <= k - tau_abs)
    s_list.append(s)
    Q_i = solver.NumVar(name=f'Q_{i}', lb=0, ub=sum(qualities))
    solver.Add(Q_i == solver.Sum([q_j * s_j for (q_j, s_j) in zip(qualities, s)]))
    solver.Add(Q_min <= Q_i)
    Q_list.append(Q_i)
solver.Maximize(Q_min)
start_time = time.perf_counter()
solver.Solve()
end_time = time.perf_counter()
print('OR-Tools (LP; min-quality objective) summed objectives:',
      round(sum(Q_i.solution_value() for Q_i in Q_list), 2))
print('OR-Tools (LP; min-quality objective) individual objectives:',
      [round(Q_i.solution_value(), 2) for Q_i in Q_list])
print('OR-Tools (LP; min-quality objective) time:', round(end_time - start_time, 2))
ortools_lp_selected = [[j for j, s_j in enumerate(s) if s_j.solution_value()] for s in s_list]

# ----- GEKKO -----
import gekko

model = gekko.GEKKO(remote=False)
model.options.SOLVER = 1  # choose MIP solver, else solutions might not be integer
model.options.WEB = 0  # do not generated a web page (doesn't really impact runtime)
model.options.DBS_LEVEL = 0  # only basic database file (doesn't really impact runtime)
s_list = []
for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
    s = [model.Var(name=f's{i}_{j}', lb=0, ub=1, integer=True) for j in range(n)]
    model.Equation(sum(s) == k)
    for s2 in s_list:
        # product_vars = []
        # for (var1, var2) in zip(s, s2):
        #     var_name = var1.name + '*' + var2.name
        #     var = model.Var(name=var_name, lb=0, ub=1, integer=True)
        #     model.Equation(var <= var1)
        #     model.Equation(var <= var2)
        #     model.Equation(1 + var >= var1 + var2)
        #     product_vars.append(var)
        # model.Equation(sum(product_vars) <= k - tau_abs)
        model.Equation(sum(s_1_j * s_2_j for (s_1_j, s_2_j) in zip(s, s2)) <= k - tau_abs)  # faster
    s_list.append(s)
objective = sum(sum(q_j * s_j for (q_j, s_j) in zip(qualities, s)) for s in s_list)
model.Maximize(objective)
start_time = time.perf_counter()
model.solve(disp=False, debug=False)
end_time = time.perf_counter()
print('GEKKO objective:', round(model.options.OBJFCNVAL, 2))
print('GEKKO time:', round(end_time - start_time, 2))  # also see model.options.SOLVETIME
gekko_selected = [[j for j, s_j in enumerate(s) if round(s_j.value[0])] for s in s_list]
