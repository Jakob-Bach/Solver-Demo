"""Optimal configuration-selection trees

Given:
- A matrix where the rows are problem instances, the columns are algorithm configurations (or just
  different algorithms), and the entries are runtimes of the configurations on the instances.
- A matrix where the rows are problem instances, the columns are features, and the entries are
  values of the features for the instances.

Goal: Build a depth-constrained balanced binary decision tree (using the features for splits) that
completely partitions the set of instances into non-overlapping subsets. Select exactly one (i.e.,
the single best) configuration for all instances in each leaf node. Optimize average runtime over
all instances, using the corresponding leaf node's configuration for each instance.

Approaches:
- MIP model that finds the globally optimal solution since it optimizes the whole tree at once.
  Scales badly, so start with depth with a depth of one and small input data (regarding number of
  instances, configurations, and features).
- A greedy search that sequentially optimizes the nodes of the tree in a top-down procedure (like
  decision trees are typically trained in machine learning). Considerably faster. Optimal for a
  depth of one but not beyond that (due to its myopic splitting procedure).
"""

import mip
import numpy as np

import time


# -- Limit data to speed up search
N_INSTANCES = 50
N_CONFIGS = 10
N_FEATURES = 5
DEPTH = 1  # of the decision tree; denotes levels of inner nodes (i.e., ignoring leaf nodes)

# -- Load data
features = np.random.default_rng().normal(size=(N_INSTANCES, N_FEATURES))
runtimes = np.random.default_rng().lognormal(size=(N_INSTANCES, N_CONFIGS))

# ---- MIP Approach

# -- Define model
model = mip.Model()
model.verbose = 0  # disable logging
N_INNER_NODES = 2 ** DEPTH - 1  # balanced binary tree
N_LEAF_NODES = 2 ** DEPTH
N_NODES = N_INNER_NODES + N_LEAF_NODES
M = float(features.max() + 1)  # used for conditional constraints in splits
SPLIT_PREC = 5  # since "mip" does not allow to model strict equalities, we round feature values to
# the number of digits defined by "SPLIT_PREC" and require the feature values in the LHS and RHS of
# a split to differ by that figure; else, instances very close to the split value might be assigned
# wrongly; in particular, instances that exactly match the split value might (randomly?) go into
# either child node instead of all being assigned to the same child; this behavior leads to
# infeasible splits and thereby overoptimistic objective values

# -- Define variables
# Indicate for each inner node if it uses a particular feature in its split:
is_node_feature = [[model.add_var(var_type=mip.BINARY)
                    for _ in range(N_FEATURES)]
                   for _ in range(N_INNER_NODES)]
# Real-valued split point for each inner node:
node_split_point = [model.add_var(var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
                    for _ in range(N_INNER_NODES)]
# Indicate for each node if a particular instance is in it:
is_node_instance = [[model.add_var(var_type=mip.BINARY)
                     for _ in range(N_INSTANCES)]
                    for _ in range(N_NODES)]
# Indicate for each leaf node if it uses a particular configuration:
is_leaf_config = [[model.add_var(var_type=mip.BINARY)
                   for _ in range(N_CONFIGS)]
                  for _ in range(N_LEAF_NODES)]
# Indicate for each leaf node if a particular instance is in it and uses a particular configuration
# (might seem redundant to prior variables, but we actually need these new variables to obtain a
# linear objective, i.e., we cannot directly multiply the "node-instance" variables with the
# "leaf-config" variables since products between variables are non-linear):
is_leaf_instance_config = [[[model.add_var(var_type=mip.BINARY)
                             for _ in range(N_CONFIGS)]
                            for _ in range(N_INSTANCES)]
                           for _ in range(N_LEAF_NODES)]

# -- Define constraints
# For each inner node, choose exactly one split feature:
for node_idx in range(N_INNER_NODES):
    model.add_constr(mip.xsum(is_node_feature[node_idx]) == 1)
# Root node contains all instances:
for instance_idx in range(N_INSTANCES):
    model.add_constr(is_node_instance[0][instance_idx] == 1)
# For each inner node, pass each contained instance to exactly one child node according to split:
for node_idx in range(N_INNER_NODES):
    for instance_idx in range(N_INSTANCES):
        # Compute value of split feature for current instance (see above for reason of rounding):
        instance_feature_value = mip.xsum(
            float(features[instance_idx, feature_idx].round(SPLIT_PREC))
            * is_node_feature[node_idx][feature_idx] for feature_idx in range(N_FEATURES))
        # If instance is in current node, it has to be in exactly one child, and vice versa (nodes
        # are numbered top-to-bottom and left-to-right, i.e., root node is "0", its children are "1"
        # and "2", children of "1" are "3" and "4", etc.):
        model.add_constr(is_node_instance[2 * node_idx + 1][instance_idx]
                         + is_node_instance[2 * node_idx + 2][instance_idx]
                         == is_node_instance[node_idx][instance_idx])
        # If instance goes to left child node, its feature value has to be <= the split value:
        model.add_constr(instance_feature_value <= node_split_point[node_idx] +
                         (1 - is_node_instance[2 * node_idx + 1][instance_idx]) * M)
        # If instance goes to right child node, its feature value has to be > the split value (since
        # strict inequalities are not allowed, we add a small epsilon and use >=):
        model.add_constr(node_split_point[node_idx] + 10 ** -SPLIT_PREC <= instance_feature_value +
                         (1 - is_node_instance[2 * node_idx + 2][instance_idx]) * M)
# For each leaf node, choose one configuration for all instances:
for leaf_idx in range(N_LEAF_NODES):
    # For each leaf node, choose exactly one configuration:
    model.add_constr(mip.xsum(is_leaf_config[leaf_idx]) == 1)
    # For each configuration, only use it for instances if it is chosen for the current leaf node:
    for config_idx in range(N_CONFIGS):
        model.add_constr(mip.xsum(is_leaf_instance_config[leaf_idx][instance_idx][config_idx]
                                  for instance_idx in range(N_INSTANCES))
                         <= N_INSTANCES * is_leaf_config[leaf_idx][config_idx])
    # For each instance, choose exactly one configuration if this instance is in current leaf:
    for instance_idx in range(N_INSTANCES):
        model.add_constr(mip.xsum(is_leaf_instance_config[leaf_idx][instance_idx][config_idx]
                                  for config_idx in range(N_CONFIGS))
                         == is_node_instance[N_INNER_NODES + leaf_idx][instance_idx])

# -- Define objective
model.objective = mip.minimize(
    mip.xsum(float(runtimes[instance_idx, config_idx]) *
             is_leaf_instance_config[leaf_idx][instance_idx][config_idx]
             for leaf_idx in range(N_LEAF_NODES)
             for instance_idx in range(N_INSTANCES)
             for config_idx in range(N_CONFIGS))
    / N_INSTANCES)

# -- Solve problem
start_time = time.perf_counter()
optimization_status = model.optimize()
end_time = time.perf_counter()
print("Optimization status (MIP):", optimization_status.name)
print("Optimal value (MIP):", round(model.objective_value, 2))
print("Optimization time (MIP):", round(end_time - start_time, 2))


# ---- Greedy Search

# Recursively search for an optimal decision tree. Inputs are a matrix of feature values (rows are
# instances, columns are features), a matrix of runtimes (rows are instances, columns are
# configurations), and the tree depth (>= 1). Output is the average runtime over all instances.
# First, find the split feature and split value that minimize the overall average runtime if the
# single best configuration is used in each of the two child nodes. Next, if the tree should have a
# depth > 1, repeat this procedure in the two child nodes. The procedure is greedy since it chooses
# individual splits without considering potential further splits down in the tree.
def optimize_tree_greedily(features: np.ndarray, runtimes: np.ndarray, depth: int = 1) -> float:
    objective = float('inf')
    split_feature = float('nan')
    split_value = float('nan')
    for current_split_feature in range(features.shape[1]):
        for current_split_value in np.unique(features[:, current_split_feature]):
            split_indicator = features[:, current_split_feature] <= current_split_value
            current_objective = (runtimes[split_indicator].sum(axis=0).min() +
                                 runtimes[~split_indicator].sum(axis=0).min()) / runtimes.shape[0]
            if current_objective < objective:
                objective = current_objective
                split_feature = current_split_feature
                split_value = current_split_value
    split_indicator = features[:, split_feature] <= split_value
    # Recursively split only if depth requires it and there are two non-empty partitions:
    if (depth > 1) and split_indicator.any() and (~split_indicator).any():
        left_objective = optimize_tree_greedily(features=features[split_indicator],
                                                runtimes=runtimes[split_indicator], depth=depth-1)
        right_objective = optimize_tree_greedily(features=features[~split_indicator],
                                                 runtimes=runtimes[~split_indicator], depth=depth-1)
        objective = (int(split_indicator.sum()) * left_objective +
                     int((~split_indicator).sum()) * right_objective) / runtimes.shape[0]
    return objective


start_time = time.perf_counter()
objective_value = optimize_tree_greedily(features=features, runtimes=runtimes, depth=DEPTH)
end_time = time.perf_counter()
print("Optimal value (greedy):", round(objective_value, 2))
print("Optimization time (greedy):", round(end_time - start_time, 2))
