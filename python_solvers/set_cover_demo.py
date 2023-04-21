"""Set Coverage

We model different variants of set coverage as MIP problems.
"""

import random
import time

import mip


# ---Prepare demo data---
# Generate random input sets of random (but limited) size
num_sets = 100
num_elements = 100
max_set_size = 10  # max size of each input set
print(f'{num_elements=}, {num_sets=}, {max_set_size=}')

random.seed(25)
set_sizes = [random.randint(1, max_set_size) for _ in range(num_sets)]
is_in_set = [random.sample([1] * set_size + [0] * (num_elements - set_size), k=num_elements)
             for set_size in set_sizes]
set_weights = [random.uniform(0, 1) for _ in range(num_sets)]
element_weights = [random.uniform(0, 1) for _ in range(num_elements)]


# ---Classical formulation of "minimum set cover"---
# Find minimum number of sets such that each element occurs in at least one selected set (problem is
# infeasible if there are elements not occurring in any of the input sets); set might be weighted
model = mip.Model()
model.verbose = 0
s_set = [model.add_var(name=f's_{i}', var_type=mip.BINARY) for i in range(num_sets)]

for j in range(num_elements):
    model.add_constr(mip.xsum(s_set[i] * is_in_set[i][j] for i in range(num_sets)) >= 1)

model.objective = mip.minimize(mip.xsum(s_set))  # set cover
# model.objective = mip.minimize(mip.xsum(s * w for s, w in zip(s_set, set_weights)))  # weighted set cover

start_time = time.perf_counter()
result = model.optimize()
end_time = time.perf_counter()
print('[Minimum set cover] Result:', result)
print('[Minimum set cover] Time:', round(end_time - start_time, 3), 's')
print('[Minimum set cover] Sets required (absolute):', round(model.objective_value))


# ---Classical formulation of "maximum coverage"---
# Find a limited number of sets such that the number of elements covered (= occuring in at least one
# selected set) is maximized; elements might be weighted, sets as well (we don't implement the
# latter; there even is a generalized version of problem where element weights are set-specific)
max_num_sets = 10

model = mip.Model()
model.verbose = 0
s_set = [model.add_var(name=f's_s_{i}', var_type=mip.BINARY) for i in range(num_sets)]
s_element = [model.add_var(name=f's_e_{j}', var_type=mip.BINARY) for j in range(num_elements)]

model.add_constr(mip.xsum(s_set) <= max_num_sets)
for j in range(num_elements):
    model.add_constr(s_element[j] <= mip.xsum(s_set[i] * is_in_set[i][j] for i in range(num_sets)))

model.objective = mip.maximize(mip.xsum(s_element))  # maximum coverage
# model.objective = mip.maximize(mip.xsum(s * w for s, w in zip(s_element, element_weights)))  # weighted maximum coverage

start_time = time.perf_counter()
result = model.optimize()
end_time = time.perf_counter()
print('[Maximum coverage] Result:', result)
print('[Maximum coverage] Time:', round(end_time - start_time, 3), 's')
print('[Maximum coverage] Elements covered (relative):',
      round(model.objective_value / num_elements, 2))


# ---Alternatives-based formulation of coverage---
# Find a limited number of sets that are dissimilar to each other such that the weights of selected
# sets are maximized
max_num_sets = 10
tau_abs = 2  # dissimilarity threshold

model = mip.Model()
model.verbose = 0
s_set = [model.add_var(name=f's_s_{i}', var_type=mip.BINARY) for i in range(num_sets)]

model.add_constr(mip.xsum(s_set) <= max_num_sets)
for i1 in range(num_sets - 1):  # compared to maximum coverage, these constraints are new
    for i2 in range(i1 + 1, num_sets):
        if sum(is_in_set[i1][j] != is_in_set[i2][j] for j in range(num_elements)) > 2 * tau_abs:
            model.add_constr(s_set[i1] + s_set[i2] <= 1)

model.objective = mip.maximize(mip.xsum(s * w for s, w in zip(s_set, set_weights)))

start_time = time.perf_counter()
result = model.optimize()
end_time = time.perf_counter()
print('[Alternatives coverage] Result:', result)
print('[Alternatives coverage] Time:', round(end_time - start_time, 3), 's')
print('[Alternatives coverage] Objective (averaged):', round(model.objective_value / max_num_sets, 2))
