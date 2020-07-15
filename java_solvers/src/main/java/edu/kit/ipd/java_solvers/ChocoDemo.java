package edu.kit.ipd.java_solvers;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.constraints.Constraint;
import org.chocosolver.solver.variables.BoolVar;

public class ChocoDemo 
{
	// Choco might introduce additional variables for the constraints, so we also need to know
	// which are the original variables (the first "numVariables" ones)
	private static long countModelsByEnumeration(Model model, int numVariables) {
		long result = 0;
		Constraint[] tempConstraints = new Constraint[numVariables];
		Constraint singleVarConstraint = null;
		String iterationString = "";
		for (long i = 0; i < Math.pow(2, numVariables); i++) {
			// Add temporary constraints (add either variable or its negation)
			iterationString = Long.toBinaryString(i);
			for (int varIdx = 0; varIdx < numVariables; varIdx++) {
				if (varIdx >= iterationString.length() || iterationString.charAt(varIdx) == '0') {
					singleVarConstraint = model.arithm((BoolVar) model.getVar(varIdx), "=", 0);
				} else {
					singleVarConstraint = model.arithm((BoolVar) model.getVar(varIdx), "=", 1);
				}
				tempConstraints[varIdx] = singleVarConstraint;
				model.post(singleVarConstraint);
			}
			// Solve
			if (model.getSolver().solve()) {
				result += 1;
			}
			// Remove temporary constraints
			model.unpost(tempConstraints);
			model.getSolver().reset();
		}
		return result;
	}
	
	// similar: model.getSolver().findAllSolutions().size();
	private static long countModelsWithSolver(Model model) {
		long result = 0;
		while (model.getSolver().solve()) {
        	result += 1;
        }
		model.getSolver().hardReset();
		return result;
	}
	
    public static void main(String[] args)
    {
        Model model = new Model("Choco Demo");
        int numVariables = 20;
        BoolVar[] variables = model.boolVarArray(numVariables);
        model.or(variables).post(); // create + add to model
        
        long startTime = System.nanoTime();
        long solutionCount = countModelsByEnumeration(model, numVariables);
        long endTime = System.nanoTime();
        System.out.println("Enumeration - Time: " + (endTime - startTime) / 1e9 +
        		" s, Solutions: " + solutionCount);
        
        startTime = System.nanoTime();
        solutionCount = countModelsWithSolver(model);
        endTime = System.nanoTime();
        System.out.println("Solving - Time: " + (endTime - startTime) / 1e9 +
        		" s, Solutions: " + solutionCount);
    }
}
