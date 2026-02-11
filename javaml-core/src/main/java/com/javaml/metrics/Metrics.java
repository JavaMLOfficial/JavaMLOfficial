package com.javaml.metrics;

import com.javaml.array.NDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Metrics for evaluating machine learning models.
 * Equivalent to scikit-learn's metrics module.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class Metrics {
    
    private Metrics() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Calculates accuracy score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @return the accuracy score
     */
    public static double accuracyScore(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        int correct = 0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            if (Math.abs(yTrue.get(i) - yPred.get(i)) < 1e-6) {
                correct++;
            }
        }
        return (double) correct / yTrue.getSize();
    }
    
    /**
     * Calculates precision score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param average the averaging method ("binary", "macro", "micro", "weighted")
     * @return the precision score
     */
    public static double precisionScore(NDArray yTrue, NDArray yPred, String average) {
        validateInputs(yTrue, yPred);
        
        if ("binary".equals(average)) {
            return precisionBinary(yTrue, yPred);
        } else if ("macro".equals(average)) {
            return precisionMacro(yTrue, yPred);
        } else {
            throw new UnsupportedOperationException("Averaging method not yet implemented: " + average);
        }
    }
    
    /**
     * Calculates recall score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param average the averaging method
     * @return the recall score
     */
    public static double recallScore(NDArray yTrue, NDArray yPred, String average) {
        validateInputs(yTrue, yPred);
        
        if ("binary".equals(average)) {
            return recallBinary(yTrue, yPred);
        } else if ("macro".equals(average)) {
            return recallMacro(yTrue, yPred);
        } else {
            throw new UnsupportedOperationException("Averaging method not yet implemented: " + average);
        }
    }
    
    /**
     * Calculates F1 score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param average the averaging method
     * @return the F1 score
     */
    public static double f1Score(NDArray yTrue, NDArray yPred, String average) {
        double precision = precisionScore(yTrue, yPred, average);
        double recall = recallScore(yTrue, yPred, average);
        
        if (precision + recall == 0) {
            return 0.0;
        }
        
        return 2.0 * (precision * recall) / (precision + recall);
    }
    
    /**
     * Calculates mean squared error.
     * 
     * @param yTrue the true values
     * @param yPred the predicted values
     * @return the mean squared error
     */
    public static double meanSquaredError(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        double sumSquaredError = 0.0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            double error = yTrue.get(i) - yPred.get(i);
            sumSquaredError += error * error;
        }
        
        return sumSquaredError / yTrue.getSize();
    }
    
    /**
     * Calculates mean absolute error.
     * 
     * @param yTrue the true values
     * @param yPred the predicted values
     * @return the mean absolute error
     */
    public static double meanAbsoluteError(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        double sumAbsoluteError = 0.0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            sumAbsoluteError += Math.abs(yTrue.get(i) - yPred.get(i));
        }
        
        return sumAbsoluteError / yTrue.getSize();
    }
    
    /**
     * Calculates R² score.
     * 
     * @param yTrue the true values
     * @param yPred the predicted values
     * @return the R² score
     */
    public static double r2Score(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        // Calculate mean of true values
        double yMean = 0.0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            yMean += yTrue.get(i);
        }
        yMean /= yTrue.getSize();
        
        // Calculate SS_res and SS_tot
        double ssRes = 0.0;
        double ssTot = 0.0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            double residual = yTrue.get(i) - yPred.get(i);
            ssRes += residual * residual;
            
            double total = yTrue.get(i) - yMean;
            ssTot += total * total;
        }
        
        if (ssTot == 0.0) {
            return 0.0;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Calculates confusion matrix.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @return a 2D array representing the confusion matrix
     */
    public static NDArray confusionMatrix(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        // Find unique classes
        Map<Double, Integer> classMap = new HashMap<>();
        int classIndex = 0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            double label = yTrue.get(i);
            if (!classMap.containsKey(label)) {
                classMap.put(label, classIndex++);
            }
        }
        for (int i = 0; i < yPred.getSize(); i++) {
            double label = yPred.get(i);
            if (!classMap.containsKey(label)) {
                classMap.put(label, classIndex++);
            }
        }
        
        int nClasses = classMap.size();
        NDArray matrix = new NDArray(nClasses, nClasses);
        
        // Count occurrences
        for (int i = 0; i < yTrue.getSize(); i++) {
            int trueClass = classMap.get(yTrue.get(i));
            int predClass = classMap.get(yPred.get(i));
            double current = matrix.get(trueClass, predClass);
            matrix.set(current + 1.0, trueClass, predClass);
        }
        
        return matrix;
    }
    
    /**
     * Calculates binary precision.
     */
    private static double precisionBinary(NDArray yTrue, NDArray yPred) {
        int tp = 0, fp = 0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            if (yPred.get(i) == 1.0 && yTrue.get(i) == 1.0) {
                tp++;
            } else if (yPred.get(i) == 1.0 && yTrue.get(i) == 0.0) {
                fp++;
            }
        }
        
        if (tp + fp == 0) {
            return 0.0;
        }
        
        return (double) tp / (tp + fp);
    }
    
    /**
     * Calculates binary recall.
     */
    private static double recallBinary(NDArray yTrue, NDArray yPred) {
        int tp = 0, fn = 0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            if (yPred.get(i) == 1.0 && yTrue.get(i) == 1.0) {
                tp++;
            } else if (yPred.get(i) == 0.0 && yTrue.get(i) == 1.0) {
                fn++;
            }
        }
        
        if (tp + fn == 0) {
            return 0.0;
        }
        
        return (double) tp / (tp + fn);
    }
    
    /**
     * Calculates macro-averaged precision.
     */
    private static double precisionMacro(NDArray yTrue, NDArray yPred) {
        // Simplified implementation
        // Full implementation would calculate per-class precision and average
        return precisionBinary(yTrue, yPred);
    }
    
    /**
     * Calculates macro-averaged recall.
     */
    private static double recallMacro(NDArray yTrue, NDArray yPred) {
        // Simplified implementation
        return recallBinary(yTrue, yPred);
    }
    
    /**
     * Validates that inputs are not null and have the same size.
     */
    private static void validateInputs(NDArray yTrue, NDArray yPred) {
        if (yTrue == null || yPred == null) {
            throw new IllegalArgumentException("Input arrays cannot be null");
        }
        if (yTrue.getSize() != yPred.getSize()) {
            throw new IllegalArgumentException(
                "yTrue and yPred must have the same size");
        }
    }
}

