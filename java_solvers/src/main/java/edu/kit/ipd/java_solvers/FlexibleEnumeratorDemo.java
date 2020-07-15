package edu.kit.ipd.java_solvers;

import edu.kit.ipd.java_solvers.expression.Or;
import edu.kit.ipd.java_solvers.expression.Problem;
import edu.kit.ipd.java_solvers.expression.Variable;

public class FlexibleEnumeratorDemo {

    public static void main(String[] args) {
        int numVariables = 20;
        Variable[] variables = new Variable[numVariables];
        for (int i = 0; i < numVariables; i++) {
            variables[i] = new Variable();
        }
        Problem problem = new Problem(variables);
        problem.addConstraint(new Or(variables));
        long startTime = System.nanoTime();
        long solutionCount = problem.countModelsByEnumeration();
        long endTime = System.nanoTime();
        System.out.println("OR - Time: " + (endTime - startTime) / 1e9 + " s, Solutions: " + solutionCount);
    }

}
