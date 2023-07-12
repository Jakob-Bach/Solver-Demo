"""Feature Selection Portfolios for Multi-Task Scenarios

Find a globally (spanning a set of tasks) relevant feature set that contains locally (task-specific)
relevant subsets. The local feature sets of size "k_local" are subsets of the global feature set of
size "k_global". The local-global subset contraints resemble a solver-portfolio scenario, where
typically one ("local") solver out of a global set of "k" ("global") solvers is chosen for each
instance. We assume univariate filter feature selection and randomly generate the feature's local
and global qualities / relevance scores.
"""

import random
import time

import mip
import z3


# ---Prepare demo data---
t = 100  # number of local feature-selection tasks
n = 100  # total number of features in the dataset
k_local = 5  # number of features to be selected per task
k_global = 10  # number of features to be selected overall

random.seed(25)
q_local = [[random.uniform(0, 1) for _ in range(n)] for _ in range(t)]  # quality for each task and feature
q_global = [random.uniform(0, 1) for _ in range(n)]  # quality for each feature


# ---MIP model---
model = mip.Model()
model.verbose = 0

s_local = [[model.add_var(name=f's_l_{i}_{j}', var_type=mip.BINARY) for j in range(n)]
           for i in range(t)]  # feature selection for each task and feature
s_global = [model.add_var(name=f's_g_{j}', var_type=mip.BINARY) for j in range(n)]

model.add_constr(mip.xsum(s_global) <= k_global)
for i in range(t):
    model.add_constr(mip.xsum(s_local[i]) <= k_local)
    for j in range(n):
        model.add_constr(s_local[i][j] <= s_global[j])
# for j in range(n):  # slower than constraining each s_local[i][j] individually
#     model.add_constr(mip.xsum(s_local[i][j] for i in range(t)) <= n * s_global[j])

local_objective = mip.xsum(q_local[i][j] * s_local[i][j] for i in range(t)
                           for j in range(n)) / (t * k_local)
global_objective = mip.xsum(q_global[j] * s_global[j] for j in range(n)) / k_global
# For unknown reasons, a weighted objective leads to much faster solution than pure local objective:
model.objective = mip.maximize(0.5 * local_objective + 0.5 * global_objective)

start_time = time.perf_counter()
result = model.optimize()
end_time = time.perf_counter()
print('Result:', result)
print('Time:', round(end_time - start_time, 3), 's')
print('Local objective:', round(local_objective.x, 2))
print('Global objective:', round(global_objective.x, 2))
print('Times selected locally:', [int(sum(s_local[i][j].x for i in range(t))) for j in range(n)])
assert sum(s.x for s in s_global) == k_global


# ---SMT model---
# way slower than MIP solution, might try t=3, n=10, k_local=2, k_global=5 to get a solution quickly
optimizer = z3.Optimize()

s_local = [[z3.Bool(name=f's_l_{i}_{j}') for j in range(n)] for i in range(t)]  # feature selection for each task and feature
s_global = [z3.Bool(name=f's_g_{j}') for j in range(n)]

optimizer.add(z3.AtMost(*s_global, k_global))
for i in range(t):
    optimizer.add(z3.AtMost(*s_local[i], k_local))
    for j in range(n):
        optimizer.add(z3.Implies(s_local[i][j], s_global[j]))

objective = optimizer.maximize(z3.Sum([q_local[i][j] * s_local[i][j] for i in range(t)
                                       for j in range(n)]) / (t * k_local))
# Optimizing a weighted combination of local and global objective is much slower than only
# optimizing local quality (note that Z3 also allows explicit multi-objective optimization):
# local_objective = z3.Real('Q_local')
# optimizer.add(local_objective == z3.Sum([q_local[i][j] * s_local[i][j] for i in range(t)
#                                           for j in range(n)]) / (t * k_local))
# global_objective = z3.Real('Q_global')
# optimizer.add(global_objective == z3.Sum([q_global[j] * s_global[j] for j in range(n)]) / k_global)
# objective = optimizer.maximize(0.5 * local_objective + 0.5 * global_objective)

start_time = time.perf_counter()
result = optimizer.check()
end_time = time.perf_counter()
print('Result:', result)
print('Time:', round(end_time - start_time, 3), 's')
print('Objective:', round(objective.value().numerator_as_long() /
                          objective.value().denominator_as_long(), 2))
# print('Local objective:', round(optimizer.model()[local_objective].numerator_as_long() /
#                                 optimizer.model()[local_objective].denominator_as_long(), 2))
# print('Global objective:', round(optimizer.model()[global_objective].numerator_as_long() /
#                                  optimizer.model()[global_objective].denominator_as_long(), 2))
print('Times selected locally:', [sum(bool(optimizer.model()[s_local[i][j]]) for i in range(t))
                                  for j in range(n)])
assert sum(bool(optimizer.model()[s]) for s in s_global) == k_global
