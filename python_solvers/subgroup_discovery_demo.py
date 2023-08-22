"""Subgroup Discovery as White-Box Problem

We model the subgroup-discovery problem, which may be solved heuristically with algorithms like
PRIM, as an exact MIP/SMT optimization problem.
"""

import time

import matplotlib.pyplot as plt
import mip
from ortools.linear_solver import pywraplp
import pandas as pd
import seaborn as sns
import sklearn.datasets
import z3


# -----Prepare demo data-----
# Dataset 1: "iris": nice for checking correctness and for visualization
# other included classification datasets: "breast_cancer", "digits", "wine"
X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
data = pd.concat((X, y.astype(str)), axis='columns')  # for sampling and plotting; else (X, y) used
# data = data.sample(n=50, random_state=25)  # to analyze scalability
# X, y = data.drop(columns='target'), data['target']

# Dataset 2: synthetically generated, which allows to vary problem size more flexibly:
# X, y = sklearn.datasets.make_classification(n_samples=100, n_features=10, n_informative=3,
#                                             n_classes=2, random_state=25)
# X, y = pd.DataFrame(X), pd.Series(y)
# data = pd.concat((X, y.astype(str)), axis='columns')

positive_class = y.unique()[0]
num_features = X.shape[1]
num_samples = X.shape[0]
num_positive_samples = int((y == positive_class).sum())
feature_minima = X.min().to_list()
feature_maxima = X.max().to_list()
feature_diff_minima = X.apply(lambda col: pd.Series(col.sort_values().unique()).diff().min())
# Define minimum difference between LHS and RHS of strict inequalities to count them as satisfied
# (actual strict inequalities typically not supported in linear optimization, so we transform
# "a < b" to "a <= b - eps"; high values deteriorate objective, low values may violate strictness):
inequality_tolerances = [0] * num_features  # one alternative: feature_diff_minima.to_list()


# -----SMT model-----
lower_bounds = [z3.Real(f'lb_{j}') for j in range(num_features)]
upper_bounds = [z3.Real(f'ub_{j}') for j in range(num_features)]
is_sample_in_box = [z3.Bool(f'y_{i}') for i in range(num_samples)]
num_positive_samples_in_box = z3.Real('n_box_pos')  # "Real" to allow float operations in objective
num_samples_in_box = z3.Real('n_box')

optimizer = z3.Optimize()
# Pick one of three objectives:
# (1) Recall, fastest objective, should be combined with an upper bound on box size (else trivial)
objective = optimizer.maximize(num_positive_samples_in_box / num_positive_samples)
# # (2) Weighted Relative Accuracy (WRacc), does not need constraint on box size
# objective = optimizer.maximize(num_positive_samples_in_box / num_samples -
#                                num_samples_in_box * num_positive_samples / (num_samples ** 2))
# # (3) Precision, should be combined with a lower bound on box size (else trivial)
# objective = optimizer.maximize(num_positive_samples_in_box / num_samples_in_box)

# Constraint type 1: Identify for each sample if it is in the subgroup's box or not
for i in range(num_samples):
    optimizer.add(z3.And([
        z3.And(float(X.iloc[i, j]) >= lower_bounds[j], float(X.iloc[i, j]) <= upper_bounds[j])
        for j in range(num_features)]) == is_sample_in_box[i])

# # Alternative formulation for in-box constraint (similar to MIP), using more variables and
# # constraints but with similar performance (split large expression in smaller sub-expressions)
# is_value_in_box = [[z3.Bool(f'x_{i}_{j}') for j in range(num_features)] for i in range(num_samples)]
# for i in range(num_samples):
#     for j in range(num_features):
#         optimizer.add(z3.And(float(X.iloc[i, j]) >= lower_bounds[j],
#                               float(X.iloc[i, j]) <= upper_bounds[j]) == is_value_in_box[i][j])
#     optimizer.add(z3.And(is_value_in_box[i]) == is_sample_in_box[i])

# Contraint type 2: Relationship between lower and upper bounds
for j in range(num_features):
    optimizer.add(lower_bounds[j] <= upper_bounds[j])

# Constraint type 3: Count samples in box
optimizer.add(num_samples_in_box == z3.Sum([z3.If(box_var, 1, 0) for box_var in is_sample_in_box]))

# Constraint type 4: Count positive samples in box
optimizer.add(num_positive_samples_in_box == z3.Sum([z3.If(box_var, 1, 0) for box_var, target
                                                     in zip(is_sample_in_box, y) if target == positive_class]))

# Constraint type 5 (objective-dependent): Upper-bound number of samples in box (else optimization
# of recall trivial; unnecessary for for WRacc; lower bound necessary for precision)
optimizer.add(num_samples_in_box <= num_positive_samples)

# # Constraint type 6 (optional): Limit number of features used, e.g., to 50% of total number
# optimizer.add(z3.Sum([z3.If(z3.Or(lower_bounds[j] > feature_minima[j],
#                                   upper_bounds[j] < feature_maxima[j]), 1, 0)
#                       for j in range(num_features)]) <= num_features * 0.5)

start_time = time.perf_counter()
result = optimizer.check()
end_time = time.perf_counter()
print('[SMT] Result:', result)
print('[SMT] Time:', round(end_time - start_time, 3), 's')

print(f'[SMT] Overall: {num_samples} samples, {num_positive_samples} positive')
print(f'[SMT] Box: {optimizer.model()[num_samples_in_box]} samples,',
      f'{optimizer.model()[num_positive_samples_in_box]} positive')
if isinstance(objective.value(), z3.IntNumRef):
    print(f'[SMT] Objective: {objective.value()}')
else:  # RatNumRef
    print('[SMT] Objective:', round(objective.value().numerator_as_long() /
                                    objective.value().denominator_as_long(), 2))
lower_bound_values = [optimizer.model()[x].numerator_as_long() /
                      optimizer.model()[x].denominator_as_long() for x in lower_bounds]
upper_bound_values = [optimizer.model()[x].numerator_as_long() /
                      optimizer.model()[x].denominator_as_long() for x in upper_bounds]

# # Create a (complete) sequence of 2D scatter plots showing the detected box:
# for j1 in range(num_features - 1):
#     for j2 in range(j1 + 1, num_features):
#         plt.figure()
#         sns.scatterplot(x=data.columns[j1], y=data.columns[j2], hue='target', data=data)
#         plt.vlines(x=(lower_bound_values[j1], upper_bound_values[j1]),
#                     ymin=lower_bound_values[j2], ymax=upper_bound_values[j2], colors='red')
#         plt.hlines(y=(lower_bound_values[j2], upper_bound_values[j2]),
#                     xmin=lower_bound_values[j1], xmax=upper_bound_values[j1], colors='red')
#         plt.show()


# -----MIP model (Python-MIP)-----
model = mip.Model()
model.verbose = 0
model.infeas_tol = 0  # disable some tolerances that may cause imprecise results
model.integer_tol = 0
model.max_mip_gap = 0
model.opt_tol = 0

# First, same variables as in SMT formulation:
lower_bounds = [model.add_var(name=f'lb_{j}', var_type=mip.CONTINUOUS, lb=feature_minima[j],
                              ub=feature_maxima[j]) for j in range(num_features)]
upper_bounds = [model.add_var(name=f'ub_{j}', var_type=mip.CONTINUOUS, lb=feature_minima[j],
                              ub=feature_maxima[j]) for j in range(num_features)]
is_sample_in_box = [model.add_var(name=f'y_{i}', var_type=mip.BINARY) for i in range(num_samples)]
num_samples_in_box = model.add_var(name='n_box', var_type=mip.INTEGER, lb=0, ub=num_samples)
num_positive_samples_in_box = model.add_var(name='n_box_pos', var_type=mip.INTEGER, lb=0,
                                            ub=num_positive_samples)
# Second, additional auxiliary variables necessary to linearize in-box constraints:
is_value_in_box_lb = [[model.add_var(name=f'box_lb_{i}_{j}', var_type=mip.BINARY)
                       for j in range(num_features)] for i in range(num_samples)]
is_value_in_box_ub = [[model.add_var(name=f'box_ub_{i}_{j}', var_type=mip.BINARY)
                       for j in range(num_features)] for i in range(num_samples)]

# Pick one of three objectives:
# (1) Recall, linear by default, should be combined with an upper bound on box size (else trivial)
model.objective = mip.maximize(num_positive_samples_in_box / num_positive_samples)
# # (2) Weighted Relative Accuracy (WRacc), linear by default, does not need constraint on box size
# model.objective = mip.maximize(num_positive_samples_in_box / num_samples -
#                                num_samples_in_box * num_positive_samples / (num_samples ** 2))
# # (3) Precision, linearized, should be combined with a lower bound on box size (else trivial)
# # For linearization, see Chang (2001): "On the polynomial mixed 0-1 fractional programming problems"
# inv_denominator = model.add_var(name='inv_d', var_type=mip.CONTINUOUS, lb=0, ub=1)
# product_vars = [model.add_var(name=f'z_{i}', var_type=mip.CONTINUOUS, lb=0, ub=1)
#                 for i in range(num_samples)]
# # Objective: "\sum_{i of positive samples} y_i * inv_d", i.e., "ordinary" numerator (number of
# # positive samples in box) times inverse of denominator, with "y_i * inv_d" linearized to "z_i"
# model.objective = mip.maximize(mip.xsum(z_i for z_i, target in zip(product_vars, y)
#                                         if target == positive_class))
# # Auxiliary constraint type 1: "\sum_i y_i * inv_d = 1", i.e., "ordinary" denominator (number of
# # samples in box) times its inverse is 1, with "y_i * inv_d" linearized to "z_i"
# model.add_constr(mip.xsum(product_vars) == 1)
# # Auxiliary constraint type 2: Linearize products "y_i * inv_d" to "z_i"
# M = 1  # "large" positive value (here, the product terms can't get very large)
# for i in range(num_samples):
#     model.add_constr(product_vars[i] <= M * is_sample_in_box[i])
#     model.add_constr(product_vars[i] <= inv_denominator)
#     model.add_constr(product_vars[i] >= inv_denominator + (is_sample_in_box[i] - 1) * M)

# Constraint type 1: Identify for each sample if it is in the subgroup's box or not
for i in range(num_samples):
    for j in range(num_features):
        # Modeling constraint satisfaction binarily: https://docs.mosek.com/modeling-cookbook/mio.html#constraint-satisfaction
        # Idea: variables (here: "is_value_in_box_lb[i][j]") express whether constraint satisfied
        M = feature_maxima[j] - feature_minima[j]  # large positive value
        m = feature_minima[j] - feature_maxima[j]  # large (in absolute terms) negative value
        model.add_constr(float(X.iloc[i, j]) + m * is_value_in_box_lb[i][j]
                         <= lower_bounds[j] - inequality_tolerances[j])  # add a small value on LHS to get < rather than <=
        model.add_constr(lower_bounds[j] <=float(X.iloc[i, j]) + M * (1 - is_value_in_box_lb[i][j]))
        model.add_constr(upper_bounds[j] + m * is_value_in_box_ub[i][j]
                         <= float(X.iloc[i, j]) - inequality_tolerances[j])
        model.add_constr(float(X.iloc[i, j]) <= upper_bounds[j] + M * (1 - is_value_in_box_ub[i][j]))
        # Modeling AND operator: https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators
        model.add_constr(is_sample_in_box[i] <= is_value_in_box_lb[i][j])
        model.add_constr(is_sample_in_box[i] <= is_value_in_box_ub[i][j])
        # third necessary constraint for AND moved outside loop and summed up over features, since
        # only simultaneous satisfaction of all LB and UB constraints implies that sample in box
    model.add_constr(mip.xsum(is_value_in_box_lb[i]) + mip.xsum(is_value_in_box_ub[i]) <=
                      is_sample_in_box[i] + 2 * num_features - 1)

# Contraint type 2: Relationship between lower and upper bounds
for j in range(num_features):
    model.add_constr(lower_bounds[j] <= upper_bounds[j])

# Constraint type 3: Count samples in box
model.add_constr(num_samples_in_box == mip.xsum(is_sample_in_box))

# Constraint type 4: Count positive samples in box
model.add_constr(num_positive_samples_in_box == mip.xsum(
    box_var for box_var, target in zip(is_sample_in_box, y) if target == positive_class))

# Constraint type 5 (objective-dependent): Upper-bound number of samples in box (here: for recall)
model.add_constr(num_samples_in_box <= num_positive_samples)

# # Constraint type 6 (optional): Limit number of features used, e.g., to 50% of total number;
# # seems to speed up the MIP optimization
# is_feature_used = [model.add_var(name='f_{j}', var_type=mip.BINARY) for j in range(num_features)]
# for j in range(num_features):
#     # model.add_constr(mip.xsum(1 - is_value_in_box_lb[i][j] for i in range(num_samples)) +
#     #                  mip.xsum(1 - is_value_in_box_ub[i][j] for i in range(num_samples)) <=
#     #                  2 * num_samples * is_feature_used[j])  # large implication slower than many small ones
#     for i in range(num_samples):
#         # If some feature values are not in the box, then this feature has a bound, i.e., is used
#         model.add_constr(1 - is_value_in_box_lb[i][j] <= is_feature_used[j])
#         model.add_constr(1 - is_value_in_box_ub[i][j] <= is_feature_used[j])
#     model.add_constr(is_feature_used[j] <=
#                       mip.xsum(1 - is_value_in_box_lb[i][j] for i in range(num_samples)) +
#                       mip.xsum(1 - is_value_in_box_ub[i][j] for i in range(num_samples)))
# model.add_constr(mip.xsum(is_feature_used) <= 0.5 * num_features)

start_time = time.perf_counter()
result = model.optimize()
end_time = time.perf_counter()
print('[Python-MIP] Result:', result)
print('[Python-MIP] Time:', round(end_time - start_time, 3), 's')

print(f'[Python-MIP] Overall: {num_samples} samples, {num_positive_samples} positive')
print(f'[Python-MIP] Box: {int(num_samples_in_box.x)} samples, {int(num_positive_samples_in_box.x)} positive')
print('[Python-MIP] Objective:', round(model.objective_value, 2))
lower_bound_values = [x.x for x in lower_bounds]
upper_bound_values = [x.x for x in upper_bounds]
# For plotting the boxes, see above


# -----MIP model (OR-Tools)-----
model = pywraplp.Solver.CreateSolver('CBC') # may also try 'SCIP' or 'CP_SAT'
model.SetNumThreads(1)

# First, same variables as in SMT formulation:
lower_bounds = [model.NumVar(name=f'lb_{j}', lb=feature_minima[j], ub=feature_maxima[j])
                for j in range(num_features)]
upper_bounds = [model.NumVar(name=f'ub_{j}', lb=feature_minima[j], ub=feature_maxima[j])
                for j in range(num_features)]
is_sample_in_box = [model.BoolVar(name=f'y_{i}') for i in range(num_samples)]
num_samples_in_box = model.IntVar(name='n_box', lb=0, ub=num_samples)
num_positive_samples_in_box = model.IntVar(name='n_box_pos', lb=0, ub=num_positive_samples)
# Second, additional auxiliary variables necessary to linearize in-box constraints:
is_value_in_box_lb = [[model.BoolVar(name=f'box_lb_{i}_{j}') for j in range(num_features)]
                      for i in range(num_samples)]
is_value_in_box_ub = [[model.BoolVar(name=f'box_ub_{i}_{j}') for j in range(num_features)]
                      for i in range(num_samples)]

# Pick one of two objectives:
# (1) Recall, linear by default, should be combined with an upper bound on box size (else trivial)
model.Maximize(num_positive_samples_in_box / num_positive_samples)
# # (2) Weighted Relative Accuracy (WRacc), linear by default, does not need constraint on box size
# model.Maximize(num_positive_samples_in_box / num_samples -
#                num_samples_in_box * num_positive_samples / (num_samples ** 2))

# Constraint type 1: Identify for each sample if it is in the subgroup's box or not
for i in range(num_samples):
    for j in range(num_features):
        # Modeling constraint satisfaction binarily: https://docs.mosek.com/modeling-cookbook/mio.html#constraint-satisfaction
        # Idea: variables (here: "is_value_in_box_lb[i][j]") express whether constraint satisfied
        M = feature_maxima[j] - feature_minima[j]  # large positive value
        m = feature_minima[j] - feature_maxima[j]  # large (in absolute terms) negative value
        model.Add(float(X.iloc[i, j]) + m * is_value_in_box_lb[i][j]
                  <= lower_bounds[j] - inequality_tolerances[j])  # add a small value on LHS to get < rather than <=
        model.Add(lower_bounds[j] <= float(X.iloc[i, j]) + M * (1 - is_value_in_box_lb[i][j]))
        model.Add(upper_bounds[j] + m * is_value_in_box_ub[i][j]
                  <= float(X.iloc[i, j]) - inequality_tolerances[j])
        model.Add(float(X.iloc[i, j]) <= upper_bounds[j] + M * (1 - is_value_in_box_ub[i][j]))
        # Modeling AND operator: https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators
        model.Add(is_sample_in_box[i] <= is_value_in_box_lb[i][j])
        model.Add(is_sample_in_box[i] <= is_value_in_box_ub[i][j])
        # third necessary constraint for AND moved outside loop and summed up over features, since
        # only simultaneous satisfaction of all LB and UB constraints implies that sample in box
    model.Add(model.Sum(is_value_in_box_lb[i]) + model.Sum(is_value_in_box_ub[i])
              <= is_sample_in_box[i] + 2 * num_features - 1)

# Contraint type 2: Relationship between lower and upper bounds
for j in range(num_features):
    model.Add(lower_bounds[j] <= upper_bounds[j])

# Constraint type 3: Count samples in box
model.Add(num_samples_in_box == model.Sum(is_sample_in_box))

# Constraint type 4: Count positive samples in box
model.Add(num_positive_samples_in_box == model.Sum(
    [box_var for box_var, target in zip(is_sample_in_box, y) if target == positive_class]))

# Constraint type 5 (objective-dependent): Upper-bound number of samples in box (here: for recall)
model.Add(num_samples_in_box <= num_positive_samples)

start_time = time.perf_counter()
result = model.Solve()
end_time = time.perf_counter()
print('[OR-Tools-MIP] Result:', result)
print('[OR-Tools-MIP] Time:', round(end_time - start_time, 3), 's')

print(f'[OR-Tools-MIP] Overall: {num_samples} samples, {num_positive_samples} positive')
print(f'[OR-Tools-MIP] Box: {int(num_samples_in_box.solution_value())} samples,',
      f'{int(num_positive_samples_in_box.solution_value())} positive')
print('[OR-Tools-MIP] Objective:', round(model.Objective().Value(), 2))
lower_bound_values = [x.solution_value() for x in lower_bounds]
upper_bound_values = [x.solution_value() for x in upper_bounds]
# For plotting the boxes, see above
