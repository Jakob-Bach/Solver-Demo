#include <chrono>
#include <iostream>
#include "z3++.h"
#include <bitset>

using namespace z3;

long count_models_with_solver(solver &solver, expr_vector &variables) {
    long result = 0;
    expr_vector alternatives(solver.ctx());
    solver.push(); //as we will add further assertions to solver, checkpoint current state
    while (solver.check() == sat) {
        result += 1;
        for (int i = 0; i < variables.size(); i++) {
            expr value = solver.get_model().get_const_interp(variables[i].decl());
            if (value) { // check if assigned at all (not null)
                if (value.is_true()) {
                    alternatives.push_back(!variables[i]);
                } else {
                    alternatives.push_back(variables[i]);
                }
            } else { // if not assigned, treat it as "false" assigned
               alternatives.push_back(variables[i]);
            }
        }
        solver.add(mk_or(alternatives));
        alternatives.resize(0); // clear()
    }
    solver.pop();
    return result;
}

// Enumeration by conditional checking (check assignment given other constraints)
long count_models_by_enumeration(solver &solver, expr_vector &variables) {
    long result = 0;
    expr_vector temp_constraints(solver.ctx());
    for (long i = 0; i < std::pow(2, variables.size()); i++) {
        long remaining_value = i;
        for (int var_idx = 0; var_idx < variables.size(); var_idx++) {
            if (remaining_value % 2 == 0) {
                temp_constraints.push_back(!variables[var_idx]);
            }
            else {
                temp_constraints.push_back(variables[var_idx]);
            }
            remaining_value = remaining_value / 2;
        }
        if (solver.check(temp_constraints) == sat) {
            result += 1;
        }
        temp_constraints.resize(0);
    }
    return result;
}

// Enumeration by substitution and simplification (assign values and check if constraints satisfied)
long count_models_by_enumeration2(solver& solver, expr_vector& variables) {
    long result = 0;
    expr_vector substitute_values(solver.ctx()); // assignment to check
    bool satisfied = true;
    for (long i = 0; i < std::pow(2, variables.size()); i++) {
        long remaining_value = i;
        for (int var_idx = 0; var_idx < variables.size(); var_idx++) {
            substitute_values.push_back(solver.ctx().bool_val(remaining_value % 2 == 1));
            remaining_value = remaining_value / 2;
        }
        satisfied = true;
        for (expr assertion : solver.assertions()) {
            if (assertion.substitute(variables, substitute_values).simplify().is_false()) {
                satisfied = false;
                break;
            }
        }
        if (satisfied) {
            result += 1;
        }
        substitute_values.resize(0);
    }
    return result;
}

int main() {
    context ctx;
    expr_vector variables(ctx);
    // Dynamic initialization does not work, variables are called "null" in that case
    /*const int numVariables = 10;
    for (int i = 0; i < numVariables; i++) {
        variables.push_back(ctx.bool_const(x[i]));
    }*/
    variables.push_back(ctx.bool_const("x0"));
    variables.push_back(ctx.bool_const("x1"));
    variables.push_back(ctx.bool_const("x2"));
    variables.push_back(ctx.bool_const("x3"));
    variables.push_back(ctx.bool_const("x4"));
    variables.push_back(ctx.bool_const("x5"));
    variables.push_back(ctx.bool_const("x6"));
    variables.push_back(ctx.bool_const("x7"));
    variables.push_back(ctx.bool_const("x8"));
    variables.push_back(ctx.bool_const("x9"));
    variables.push_back(ctx.bool_const("x10"));
    variables.push_back(ctx.bool_const("x11"));
    variables.push_back(ctx.bool_const("x12"));
    variables.push_back(ctx.bool_const("x13"));
    variables.push_back(ctx.bool_const("x14"));
    variables.push_back(ctx.bool_const("x15"));
    variables.push_back(ctx.bool_const("x16"));
    variables.push_back(ctx.bool_const("x17"));
    variables.push_back(ctx.bool_const("x18"));
    variables.push_back(ctx.bool_const("x19"));
    solver solver(ctx);
    solver.add(mk_and(variables));

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    long solutionCount = count_models_by_enumeration(solver, variables);
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    std::cout << "Enumeration (conditional check) - Time: " << (endTime - startTime).count() / 1e9
        << " s, Solutions: " << solutionCount << "\n";

    startTime = std::chrono::steady_clock::now();
    solutionCount = count_models_by_enumeration2(solver, variables);
    endTime = std::chrono::steady_clock::now();
    std::cout << "Enumeration (substitute + simplify) - Time: " << (endTime - startTime).count() / 1e9
        << " s, Solutions: " << solutionCount << "\n";

    startTime = std::chrono::steady_clock::now();
    solutionCount = count_models_with_solver(solver, variables);
    endTime = std::chrono::steady_clock::now();
    std::cout << "Solving - Time: " << (endTime - startTime).count() / 1e9 << " s, Solutions: " << solutionCount;
}
