package edu.kit.ipd.java_solvers;

import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverSolutionCallback;
import com.google.ortools.sat.CpSolverStatus;
import com.google.ortools.sat.IntVar;
import com.google.ortools.sat.Literal;

public class ORToolsDemo {

    static {
        System.loadLibrary("jniortools"); // contained in binary distribution
    }
    
    // Attention: modifies the input "model"
    private static long countModelsWithSolver(CpModel model, IntVar[] variables) {
        long result = 0;
        Literal[] alternative = new Literal[variables.length];
        CpSolver solver = new CpSolver();
        while(solver.solve(model) == CpSolverStatus.FEASIBLE) {
            result += 1;
            for (int i = 0; i < variables.length; i++) {
                if (solver.booleanValue(variables[i])) {
                    alternative[i] = variables[i].not();
                } else {
                    alternative[i] = variables[i];
                }
            }
            model.addBoolOr(alternative);
        }
        return result;
    }
    
    // Helper class for the native enumeration of all solutions
    private static class SolutionCounter extends CpSolverSolutionCallback  {
        
        private long solutionCount = 0;
        
        @Override
        public void onSolutionCallback() {
            this.solutionCount += 1;
        }
        
        public long solutionCount() {
            return this.solutionCount;
        }
    }
    
    private static long countModelsNatively(CpModel model) {
        CpSolver solver = new CpSolver();
        SolutionCounter counterCallback = new SolutionCounter();
        solver.searchAllSolutions(model, counterCallback);
        return counterCallback.solutionCount();
    }
    
    public static void main(String[] args) {
        CpModel model = new CpModel();
        int numVariables = 10;
        IntVar[] variables = new IntVar[numVariables];
        for (int i = 0; i < numVariables; i++) {
            variables[i] = model.newBoolVar("x" + i);
        }
        model.addBoolAnd(variables);

        long startTime = System.nanoTime();
        long solutionCount = countModelsNatively(model);
        long endTime = System.nanoTime();
        System.out.println("Native counting - Time: " + (endTime - startTime) / 1e9 +
                " s, Solutions: " + solutionCount);
        
        startTime = System.nanoTime();
        solutionCount = countModelsWithSolver(model, variables);
        endTime = System.nanoTime();
        System.out.println("Solving - Time: " + (endTime - startTime) / 1e9 +
                " s, Solutions: " + solutionCount);
    }

}
