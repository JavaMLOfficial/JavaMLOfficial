package com.javaml.linear;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.HashMap;
import java.util.Map;

/**
 * Stochastic Gradient Descent Regressor.
 * Equivalent to scikit-learn's SGDRegressor.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class SGDRegressor extends BaseEstimator implements Estimator {
    
    private double learningRate;
    private int maxIterations;
    private double tolerance;
    private double alpha; // L2 regularization
    private NDArray coefficients;
    private double intercept;
    private boolean fitted = false;
    
    /**
     * Creates a new SGDRegressor with default parameters.
     */
    public SGDRegressor() {
        this(0.01, 1000, 1e-3, 0.0001);
    }
    
    /**
     * Creates a new SGDRegressor with specified parameters.
     * 
     * @param learningRate the learning rate
     * @param maxIterations maximum number of iterations
     * @param tolerance convergence tolerance
     * @param alpha L2 regularization parameter
     */
    public SGDRegressor(double learningRate, int maxIterations, 
                        double tolerance, double alpha) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.alpha = alpha;
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
        
        // Initialize coefficients
        coefficients = new NDArray(nFeatures);
        intercept = 0.0;
        
        // Stochastic gradient descent
        for (int iter = 0; iter < maxIterations; iter++) {
            double previousLoss = computeLoss(X, y);
            
            // Shuffle data
            int[] indices = new int[nSamples];
            for (int i = 0; i < nSamples; i++) {
                indices[i] = i;
            }
            shuffle(indices);
            
            // Update for each sample
            for (int idx : indices) {
                // Compute prediction
                double prediction = intercept;
                for (int j = 0; j < nFeatures; j++) {
                    prediction += coefficients.get(j) * X.get(idx, j);
                }
                
                // Compute error
                double error = prediction - y.get(idx);
                
                // Update intercept
                intercept -= learningRate * error;
                
                // Update coefficients with L2 regularization
                for (int j = 0; j < nFeatures; j++) {
                    double gradient = error * X.get(idx, j) + alpha * coefficients.get(j);
                    double newCoeff = coefficients.get(j) - learningRate * gradient;
                    coefficients.set(newCoeff, j);
                }
            }
            
            // Check convergence
            double currentLoss = computeLoss(X, y);
            if (Math.abs(previousLoss - currentLoss) < tolerance) {
                break;
            }
        }
        
        fitted = true;
    }
    
    /**
     * Computes mean squared error loss.
     */
    private double computeLoss(NDArray X, NDArray y) {
        int nSamples = X.getShape()[0];
        int nFeatures = X.getShape()[1];
        
        double loss = 0.0;
        for (int i = 0; i < nSamples; i++) {
            double prediction = intercept;
            for (int j = 0; j < nFeatures; j++) {
                prediction += coefficients.get(j) * X.get(i, j);
            }
            double error = prediction - y.get(i);
            loss += error * error;
        }
        loss /= nSamples;
        
        // Add L2 regularization
        double reg = 0.0;
        for (int j = 0; j < nFeatures; j++) {
            reg += coefficients.get(j) * coefficients.get(j);
        }
        loss += alpha * reg;
        
        return loss;
    }
    
    /**
     * Shuffles an array.
     */
    private void shuffle(int[] array) {
        java.util.Random random = new java.util.Random();
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    @Override
    public NDArray predict(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        return predictInternal(X);
    }
    
    /**
     * Internal prediction method.
     */
    private NDArray predictInternal(NDArray X) {
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
        
        double[] predictions = new double[nSamples];
        for (int i = 0; i < nSamples; i++) {
            double sum = intercept;
            for (int j = 0; j < nFeatures; j++) {
                sum += coefficients.get(j) * X.get(i, j);
            }
            predictions[i] = sum;
        }
        
        return new NDArray(predictions, predictions.length);
    }
    
    @Override
    public double score(NDArray X, NDArray y) {
        NDArray predictions = predict(X);
        
        // Calculate R²
        double ssRes = 0.0;
        double ssTot = 0.0;
        double yMean = 0.0;
        
        for (int i = 0; i < y.getSize(); i++) {
            yMean += y.get(i);
        }
        yMean /= y.getSize();
        
        for (int i = 0; i < y.getSize(); i++) {
            double residual = y.get(i) - predictions.get(i);
            ssRes += residual * residual;
            double total = y.get(i) - yMean;
            ssTot += total * total;
        }
        
        return 1.0 - (ssRes / ssTot);
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
        params.put("tolerance", tolerance);
        params.put("alpha", alpha);
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
        if (params.containsKey("tolerance")) {
            this.tolerance = ((Number) params.get("tolerance")).doubleValue();
        }
        if (params.containsKey("alpha")) {
            this.alpha = ((Number) params.get("alpha")).doubleValue();
        }
    }
}

