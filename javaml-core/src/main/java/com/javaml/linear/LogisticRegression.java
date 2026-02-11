package com.javaml.linear;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.HashMap;
import java.util.Map;

/**
 * Logistic Regression classifier.
 * Equivalent to scikit-learn's LogisticRegression.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class LogisticRegression extends BaseEstimator implements Estimator {
    
    private NDArray coefficients;
    private double intercept;
    private double learningRate;
    private int maxIterations;
    private boolean fitted = false;
    
    /**
     * Creates a new LogisticRegression with default parameters.
     */
    public LogisticRegression() {
        this(0.01, 1000);
    }
    
    /**
     * Creates a new LogisticRegression with specified parameters.
     * 
     * @param learningRate the learning rate for gradient descent
     * @param maxIterations the maximum number of iterations
     */
    public LogisticRegression(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
    }
    
    @Override
    public void fit(NDArray X, NDArray y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D (samples x features)");
        }
        if (y.getNdims() != 1) {
            throw new IllegalArgumentException("y must be 1D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nSamples != y.getSize()) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        // Initialize coefficients and intercept
        coefficients = new NDArray(nFeatures);
        intercept = 0.0;
        
        // Gradient descent
        for (int iter = 0; iter < maxIterations; iter++) {
            // Calculate predictions (sigmoid)
            NDArray predictions = predictProbability(X);
            
            // Calculate gradients
            double[] gradCoeff = new double[nFeatures];
            double gradIntercept = 0.0;
            
            for (int i = 0; i < nSamples; i++) {
                double error = predictions.get(i) - y.get(i);
                gradIntercept += error;
                
                for (int j = 0; j < nFeatures; j++) {
                    gradCoeff[j] += error * X.get(i, j);
                }
            }
            
            // Update coefficients
            intercept -= learningRate * gradIntercept / nSamples;
            for (int j = 0; j < nFeatures; j++) {
                double current = coefficients.get(j);
                coefficients.set(current - learningRate * gradCoeff[j] / nSamples, j);
            }
        }
        
        fitted = true;
    }
    
    @Override
    public NDArray predict(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D");
        }
        
        NDArray probabilities = predictProbability(X);
        double[] predictions = new double[probabilities.getSize()];
        
        for (int i = 0; i < probabilities.getSize(); i++) {
            predictions[i] = probabilities.get(i) >= 0.5 ? 1.0 : 0.0;
        }
        
        return new NDArray(predictions, predictions.length);
    }
    
    /**
     * Predicts class probabilities.
     * 
     * @param X the input features
     * @return the predicted probabilities
     */
    public NDArray predictProbability(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nFeatures != coefficients.getSize()) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted model");
        }
        
        double[] probabilities = new double[nSamples];
        for (int i = 0; i < nSamples; i++) {
            double z = intercept;
            for (int j = 0; j < nFeatures; j++) {
                z += coefficients.get(j) * X.get(i, j);
            }
            probabilities[i] = sigmoid(z);
        }
        
        return new NDArray(probabilities, probabilities.length);
    }
    
    @Override
    public double score(NDArray X, NDArray y) {
        NDArray predictions = predict(X);
        return calculateAccuracy(predictions, y);
    }
    
    /**
     * Sigmoid function.
     */
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    /**
     * Calculates accuracy.
     */
    private double calculateAccuracy(NDArray predictions, NDArray y) {
        int correct = 0;
        for (int i = 0; i < predictions.getSize(); i++) {
            if (Math.abs(predictions.get(i) - y.get(i)) < 1e-6) {
                correct++;
            }
        }
        return (double) correct / predictions.getSize();
    }
    
    /**
     * Gets the coefficients.
     * 
     * @return the coefficients
     */
    public NDArray getCoefficients() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return new NDArray(coefficients);
    }
    
    /**
     * Gets the intercept.
     * 
     * @return the intercept
     */
    public double getIntercept() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return intercept;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("learning_rate", learningRate);
        params.put("max_iterations", maxIterations);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("learning_rate")) {
            this.learningRate = ((Number) params.get("learning_rate")).doubleValue();
        }
        if (params.containsKey("max_iterations")) {
            this.maxIterations = ((Number) params.get("max_iterations")).intValue();
        }
    }
}

