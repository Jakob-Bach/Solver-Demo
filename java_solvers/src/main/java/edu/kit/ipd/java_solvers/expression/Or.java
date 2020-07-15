package edu.kit.ipd.java_solvers.expression;

public class Or implements Expression {

private Expression[] expressions;
    
    public Or(Expression[] expressions) {
        this.expressions = expressions;
    }
    
    public boolean isTrue() {
        for (Expression expression: this.expressions) {
            if (expression.isTrue()) {
                return true;
            }
        }
        return false;
    }

}
