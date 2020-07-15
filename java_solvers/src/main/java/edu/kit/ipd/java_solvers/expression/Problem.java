package edu.kit.ipd.java_solvers.expression;

import java.util.Collection;
import java.util.LinkedList;

public class Problem {

    Variable[] variables;
    Collection<Expression> constraints;
    
    public Problem(Variable[] variables) {
        this.variables = variables;
        this.constraints = new LinkedList<Expression>();
    }
    
    public void addConstraint(Expression constraint) {
        this.constraints.add(constraint);
    }
    
    public long countModelsByEnumeration() {
        long result = 0;
        Assignment: for (long i = 0; i < Math.pow(2, this.variables.length); i++) {
            // Assign
            long remainingValue = i;
            for (Variable variable: this.variables) {
                variable.value = remainingValue % 2 == 1;
                remainingValue /= 2;
            }
            // Check SAT
            for (Expression constraint: this.constraints) {
                if (!constraint.isTrue()) {
                    continue Assignment;
                }
            }
            result += 1;
        }
        return result;
    }
    
}
