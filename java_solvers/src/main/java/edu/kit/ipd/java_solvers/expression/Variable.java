package edu.kit.ipd.java_solvers.expression;

public class Variable implements Expression {

    public boolean value;
    
    public boolean isTrue() {
        return this.value;
    }

}
