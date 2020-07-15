package edu.kit.ipd.java_solvers;

import com.microsoft.z3.BoolExpr;
import com.microsoft.z3.Context;
import com.microsoft.z3.Expr;
import com.microsoft.z3.Solver;
import com.microsoft.z3.Status;

public class Z3Demo {

    private static long countModelsWithSolver(Solver solver, BoolExpr[] variables, Context ctx) {
        long result = 0;
        BoolExpr[] alternative = new BoolExpr[variables.length];
        solver.push(); //as we will add further assertions to solver, checkpoint current state
        while (solver.check() == Status.SATISFIABLE) {
            result += 1;
            for (int i = 0; i < variables.length; i++) {
                Expr value = solver.getModel().getConstInterp(variables[i]);
                if (value != null && value.isTrue()) { // some variables might not be initialized (null)
                    alternative[i] = ctx.mkNot(variables[i]);
                } else {
                    alternative[i] = variables[i];
                }
            }
            solver.add(ctx.mkOr(alternative)); // at least one variable different
        }
        solver.pop();
        return result;
    }
    
    // Enumeration by conditional checking (check assignment given other constraints)
    private static long countModelsByEnumeration(Solver solver, BoolExpr[] variables, Context ctx) {
        long result = 0;
        BoolExpr[] tempConstraints = new BoolExpr[variables.length];
        String iterationString = "";
        for (long i = 0; i < Math.pow(2, variables.length); i++) {
            // Add temporary constraints (add either variable or its negation)
            iterationString = Long.toBinaryString(i);
            for (int varIdx = 0; varIdx < variables.length; varIdx++) {
                if (varIdx >= iterationString.length() || iterationString.charAt(varIdx) == '0') {
                    tempConstraints[varIdx] = ctx.mkNot(variables[varIdx]);
                } else {
                    tempConstraints[varIdx] = variables[varIdx];
                }
            }
            if (solver.check(tempConstraints) == Status.SATISFIABLE) {
                result += 1;
            }
        }
        return result;
    }
    
    // Enumeration by substitution and simplification (assign values and check if constraints satisfied)
    private static long countModelsByEnumeration2(Solver solver, BoolExpr[] variables, Context ctx) {
        long result = 0;
        BoolExpr[] substituteValues = new BoolExpr[variables.length]; // assignment to check
        String iterationString = "";
        boolean satisfied = true;
        for (long i = 0; i < Math.pow(2, variables.length); i++) {
            iterationString = Long.toBinaryString(i);
            for (int varIdx = 0; varIdx < variables.length; varIdx++) {
                substituteValues[varIdx] = ctx.mkBool(varIdx < iterationString.length() &&
                        iterationString.charAt(varIdx) == '1');
            }
            satisfied = true;
            for (BoolExpr assertion: solver.getAssertions()) {
                if (assertion.substitute(variables, substituteValues).simplify().isFalse()) {
                    satisfied = false;
                    break;
                }
            }
            if (satisfied) {
                result += 1;
            }
        }
        return result;
    }
    
    public static void main(String[] args) {
        Context ctx = new Context();
        int numVariables = 10;
        BoolExpr[] variables = new BoolExpr[numVariables];
        for (int i = 0; i < numVariables; i++) {
            variables[i] = ctx.mkBoolConst("x" + i);
        }
        Solver solver = ctx.mkSolver();
        solver.add(ctx.mkOr(variables));
        
        long startTime = System.nanoTime();
        long solutionCount = countModelsByEnumeration(solver, variables, ctx);
        long endTime = System.nanoTime();
        System.out.println("Enumeration (conditional check) - Time: " + (endTime - startTime) / 1e9 +
                " s, Solutions: " + solutionCount);
        
        startTime = System.nanoTime();
        solutionCount = countModelsByEnumeration2(solver, variables, ctx);
        endTime = System.nanoTime();
        System.out.println("Enumeration (substitute + simplify) - Time: " + (endTime - startTime) / 1e9 +
                " s, Solutions: " + solutionCount);
        
        startTime = System.nanoTime();
        solutionCount = countModelsWithSolver(solver, variables, ctx);
        endTime = System.nanoTime();
        System.out.println("Solving - Time: " + (endTime - startTime) / 1e9 +
                " s, Solutions: " + solutionCount);
        
        ctx.close();
    }

}
