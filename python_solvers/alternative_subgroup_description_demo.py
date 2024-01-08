"""Find Alternative Subgroup Descriptions

Search for subgroup descriptions with at most "k" features that contain at least a certain number
"tau_abs" of features not used in a given ("original") subgroup description but still try to
replicate the prediction (= subgroup membership of instances) of the former as good as possible
(optimizing the Hamming similarity of binary predictions).
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.datasets
import z3


# -----User parameters-----

k = 3  # maximum number of features used (= restricted) in each subgroup description
tau_abs = 2  # min number of new features used in alternative subgroup description
a = 10  # number of alternatives

# -----Prepare demo data-----

X, y = sklearn.datasets.load_wine(as_frame=True, return_X_y=True)
y = (y == y[0]).astype(int)  # binarize target

n_instances = X.shape[0]
n_features = X.shape[1]
n_pos_instances = y.sum()
feature_minima = X.min().to_list()
feature_maxima = X.max().to_list()


# -----Functions-----

# Find original subgroup or (if optional arguments are passed) an alternative subgroup description
# for dataset (X, y) from global scope. Return bounds of subgroup, indicators of instances'
# subgroup memberships, indicators of feature' usage in subgroup, and statistics.
# - "is_feature_used_list": for each existing subgroup and each feature, indicate if used
# - "is_instance_in_box": for each instance, indicate if in existing subgroup
def optimize(is_feature_used_list: Optional[List[List[bool]]] = None,
             is_instance_in_box: Optional[List[bool]] = None) -> Dict[str, Any]:
    is_alternative = (is_feature_used_list is not None) and (is_instance_in_box is not None)

    # Define variables of the optimization problem:
    lb_vars = [z3.Real(f'lb_{j}') for j in range(n_features)]
    ub_vars = [z3.Real(f'ub_{j}') for j in range(n_features)]
    is_instance_in_box_vars = [z3.Bool(f'x_{i}') for i in range(n_instances)]
    is_feature_used_vars = [z3.Bool(f'f_{j}') for j in range(n_features)]

    # Define auxiliary expressions for use in objective and potentially even constraints
    # (could also be variables, bound by "==" constraints; roughly same optimizer performance):
    n_instances_in_box = z3.Sum([z3.If(box_var, 1.0, 0) for box_var in is_instance_in_box_vars])
    n_pos_instances_in_box = z3.Sum([z3.If(box_var, 1.0, 0) for box_var, target
                                     in zip(is_instance_in_box_vars, y) if target == 1])

    # Define optimizer and objective:
    optimizer = z3.Optimize()
    if is_alternative:  # optimize Hamming similarity to previous instance-in-box values
        objective = optimizer.maximize(z3.Sum([z3.If(var == val, 1, 0) for var, val in
                                               zip(is_instance_in_box_vars, is_instance_in_box)]
                                              ) * 1.0 / n_instances)  # * 1.0 necessary for float
    else:  # optimize WRAcc
        objective = optimizer.maximize(n_pos_instances_in_box / n_instances -
                                       n_instances_in_box * n_pos_instances / (n_instances ** 2))

    # Define constraints:
    # (1) Identify for each instance if it is in the subgroup's box or not
    for i in range(n_instances):
        optimizer.add(is_instance_in_box_vars[i] ==
                      z3.And([z3.And(float(X.iloc[i, j]) >= lb_vars[j],
                                     float(X.iloc[i, j]) <= ub_vars[j])
                              for j in range(n_features)]))
    # (2) Relationship between lower and upper bound for each feature
    for j in range(n_features):
        optimizer.add(lb_vars[j] <= ub_vars[j])
    # (3) Limit number of features used in the box (i.e., where bounds exclude instances)
    for j in range(n_features):
        optimizer.add(is_feature_used_vars[j] == z3.Or(lb_vars[j] > feature_minima[j],
                                                       ub_vars[j] < feature_maxima[j]))
    optimizer.add(z3.Sum([z3.If(is_feature_used, 1, 0)
                          for is_feature_used in is_feature_used_vars]) <= k)
    # (4) Make alternatives use a certain number of new features (not used in other subgroups)
    if is_alternative:
        for is_feature_used_values in is_feature_used_list:
            optimizer.add(z3.Sum([is_feature_used_vars[j] for j in range(n_features)
                                  if not is_feature_used_values[j]]) >= tau_abs)

    # Optimize:
    start_time = time.perf_counter()
    optimization_status = optimizer.check()
    end_time = time.perf_counter()

    # Prepare_results:
    if isinstance(objective.value(), z3.IntNumRef):
        objective_value = float(objective.value().as_long())
    else:  # RatNumRef
        objective_value = float(objective.value().numerator_as_long() /
                                objective.value().denominator_as_long())
    is_instance_in_box = [bool(optimizer.model()[var]) for var in is_instance_in_box_vars]
    is_feature_used = [bool(optimizer.model()[var]) for var in is_feature_used_vars]
    box_lbs = X.iloc[is_instance_in_box].min()
    box_ubs = X.iloc[is_instance_in_box].max()

    # Return results:
    return {'optimization_status': str(optimization_status),
            'optimization_time': end_time - start_time,
            'objective_value': objective_value,
            'box_lbs': box_lbs,
            'box_ubs': box_ubs,
            'is_instance_in_box': is_instance_in_box,
            'is_feature_used': is_feature_used}


# Compute weighted relative accuracy for passed subgroups' bounds and dataset from global scope.
def wracc(box_lbs: pd.Series, box_ubs: pd.Series) -> float:
    y_pred = pd.Series((X.ge(box_lbs) & X.le(box_ubs)).all(axis='columns').astype(int),
                       index=X.index)
    n_true_pos = (y & y_pred).sum()
    n_instances = len(y)
    n_actual_pos = y.sum()
    n_pred_pos = y_pred.sum()
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


# Compute Jaccard similarity between two subgroups based on their instance-membership indicators.
def jaccard(is_instance_in_box_1: List[bool], is_instance_in_box_2: List[bool]) -> float:
    size_intersection = sum([x and y for x, y in zip(is_instance_in_box_1, is_instance_in_box_2)])
    size_union = sum([x or y for x, y in zip(is_instance_in_box_1, is_instance_in_box_2)])
    return size_intersection / size_union


# -----Evaluation-----

is_feature_used_list = []  # alternatives should use new feature relative to all prior subgroups
results = []
for i in range(a + 1):
    if i == 0:  # each alternative should try to replicate prediction of this "orignal" subgroup
        result = optimize()
        is_instance_in_box = result['is_instance_in_box']
    else:
        result = optimize(is_feature_used_list=is_feature_used_list,
                          is_instance_in_box=is_instance_in_box)
    is_feature_used_list.append(result['is_feature_used'])
    result['wracc'] = wracc(box_lbs=result['box_lbs'], box_ubs=result['box_ubs'])
    result['jaccard'] = jaccard(is_instance_in_box_1=is_instance_in_box,
                                is_instance_in_box_2=result['is_instance_in_box'])
    results.append(result)
results = pd.DataFrame(results)
print('Total number of features used:', np.array(is_feature_used_list).any(axis=0).sum())
print(results[['optimization_status', 'optimization_time',
               'objective_value', 'wracc', 'jaccard']].round(3))
# "objective_value" is WRACC for 0th subgroup and Hamming similarity for all other subgroups
