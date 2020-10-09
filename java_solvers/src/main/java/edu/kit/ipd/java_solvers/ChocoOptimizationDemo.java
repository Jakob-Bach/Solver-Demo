package edu.kit.ipd.java_solvers;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;

import java.util.Random;

// Simple optimization problem with boolean variables, linear target function and
// one cardinality constraint (a special case of a knapsack problem)
public class ChocoOptimizationDemo {

    public static void main(String[] args) {
        int n = 20;
        int k = 10;
        Model model = new Model("Choco Optimization Demo");
        BoolVar[] variables = model.boolVarArray(n);
        int[] utilities = new int[n]; // to define weighted sum as objective
        Random rng = new Random();
        rng.setSeed(25);
        for (int i = 0; i < n; i++) {
            utilities[i] = (int) (100 * rng.nextDouble());
        }
        IntVar objectiveVar = model.intVar("objective", 0, 100 * n); // need to specify lower and upper bound
        model.scalar(variables, utilities, "=", objectiveVar).post(); // weighted sum
        model.setObjective(Model.MAXIMIZE, objectiveVar); // objective represented by a variable
        model.sum(variables, "<=", k).post(); // cardinality constraint

        Solution bestModel = new Solution(model); // to save best solution; else, we don't get assignments
        int iterationCount = 0;
        long startTime = System.nanoTime();
        while(model.getSolver().solve()) { // solve() returns just one valid solution
            System.out.print("."); // cheap progress bar
            iterationCount++;
            bestModel.record(); // save current solution, delete previous
        }
        long endTime = System.nanoTime();
        System.out.println();

        System.out.println("Time: " + (endTime - startTime) / 1e9 + " s");
        System.out.println("Number of iterations: " + iterationCount);
        System.out.println("Objective value: " + model.getSolver().getBestSolutionValue()); // or: objectiveVar.getValue()
        System.out.print("(Utilities, Assignments): ");
        for (int i = 0; i < n; i++) {
            System.out.print("(" + utilities[i] + "," + bestModel.getIntVal(variables[i]) + ") ");
        }
        System.out.println();
    }

}
