package edu.kit.ipd.java_solvers;

public class ArithmeticEnumerationDemo {

    private static long countOrModelsByEnumeration(int numVariables) {
        long result = 0;
        Assignment: for (long i = 0; i < Math.pow(2, numVariables); i++) {
            long remainingValue = i;
            for (int varIdx = 0; varIdx < numVariables; varIdx++) {
               if (remainingValue % 2 == 1) {
                   result += 1;
                   continue Assignment; // early abandoning: only one variable has to be true
               }
               remainingValue /= 2;
            }
        }
        return result;
    }

    // Slower version which uses built-in int-to-binary-conversion
    private static long countOrModelsByEnumeration2(int numVariables) {
        long result = 0;
        String iterationString = "";
        Assignment: for (long i = 0; i < Math.pow(2, numVariables); i++) {
            iterationString = Long.toBinaryString(i);
            for (int varIdx = 0; varIdx < numVariables; varIdx++) {
                if (varIdx < iterationString.length() && iterationString.charAt(varIdx) == '1') {
                    result += 1;
                    continue Assignment;
                }
            }
        }
        return result;
    }

    private static long countAndModelsByEnumeration(int numVariables) {
        long result = 0;
        Assignment: for (long i = 0; i < Math.pow(2, numVariables); i++) {
            long remainingValue = i;
            for (int varIdx = 0; varIdx < numVariables; varIdx++) {
               if (remainingValue % 2 == 0) {
                   continue Assignment; // early abandoning: all variables have to be true
               }
               remainingValue /= 2;
            }
            result += 1;
        }
        return result;
    }

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        long solutionCount = countOrModelsByEnumeration(20);
        long endTime = System.nanoTime();
        System.out.println("OR - Time: " + (endTime - startTime) / 1e9 + " s, Solutions: " + solutionCount);

        startTime = System.nanoTime();
        solutionCount = countAndModelsByEnumeration(20);
        endTime = System.nanoTime();
        System.out.println("AND - Time: " + (endTime - startTime) / 1e9 + " s, Solutions: " + solutionCount);
    }

}
