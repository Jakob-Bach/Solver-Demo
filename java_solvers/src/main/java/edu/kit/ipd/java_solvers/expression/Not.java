package edu.kit.ipd.java_solvers.expression;

public class Not implements Expression {

    private Expression expression;
    
    public Not(Expression expression) {
        this.expression = expression;
    }
    
    public boolean isTrue() {
        return !expression.isTrue();
    }

}
