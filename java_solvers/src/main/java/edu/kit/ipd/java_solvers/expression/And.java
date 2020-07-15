package edu.kit.ipd.java_solvers.expression;

public class And implements Expression {

    private Expression[] expressions;
    
    public And(Expression[] expressions) {
        this.expressions = expressions;
    }
    
    public boolean isTrue() {
        for (Expression expression: this.expressions) {
            if (!expression.isTrue()) {
                return false;
            }
        }
        return true;
    }

}
