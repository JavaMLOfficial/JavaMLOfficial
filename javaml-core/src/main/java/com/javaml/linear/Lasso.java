package com.javaml.linear;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.HashMap;
import java.util.Map;

/**
 * Lasso regression with L1 regularization.
 * Equivalent to scikit-learn's Lasso.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class Lasso extends BaseEstimator implements Estimator {
    
    private double alpha;
    private int maxIterations;
    private double tolerance;
    private NDArray coefficients;
    private double intercept;
    private boolean fitted = false;
    
    /**
     * Creates a new Lasso regressor with default alpha=1.0.
     */
    public Lasso() {
        this(1.0, 1000, 1e-4);
    }
    
    /**
     * Creates a new Lasso regressor with specified parameters.
     * 
     * @param alpha regularization strength
     * @param maxIterations maximum number of iterations
     * @param tolerance convergence tolerance
     */
    public Lasso(double alpha, int maxIterations, double tolerance) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
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
        
        // Coordinate descent for Lasso
        double learningRate = 0.01;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            double maxChange = 0.0;
            
            // Update intercept
            double interceptGrad = 0.0;
            for (int i = 0; i < nSamples; i++) {
                double prediction = intercept;
                for (int j = 0; j < nFeatures; j++) {
                    prediction += coefficients.get(j) * X.get(i, j);
                }
                interceptGrad += (prediction - y.get(i));
            }
            double oldIntercept = intercept;
            intercept -= learningRate * interceptGrad / nSamples;
            maxChange = Math.max(maxChange, Math.abs(intercept - oldIntercept));
            
            // Update each coefficient with soft thresholding (L1 regularization)
            for (int j = 0; j < nFeatures; j++) {
                double grad = 0.0;
                for (int i = 0; i < nSamples; i++) {
                    double prediction = intercept;
                    for (int k = 0; k < nFeatures; k++) {
                        prediction += coefficients.get(k) * X.get(i, k);
                    }
                    grad += (prediction - y.get(i)) * X.get(i, j);
                }
                grad /= nSamples;
                
                double oldCoeff = coefficients.get(j);
                double newCoeff = softThreshold(oldCoeff - learningRate * grad, alpha * learningRate);
                coefficients.set(newCoeff, j);
                maxChange = Math.max(maxChange, Math.abs(newCoeff - oldCoeff));
            }
            
            if (maxChange < tolerance) {
                break;
            }
        }
        
        fitted = true;
    }
    
    /**
     * Soft thresholding operator for L1 regularization.
     */
    private double softThreshold(double value, double threshold) {
        if (value > threshold) {
            return value - threshold;
        } else if (value < -threshold) {
            return value + threshold;
        } else {
            return 0.0;
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
        params.put("alpha", alpha);
        params.put("max_iterations", maxIterations);
        params.put("tolerance", tolerance);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) {
            this.alpha = ((Number) params.get("alpha")).doubleValue();
        }
        if (params.containsKey("max_iterations")) {
            this.maxIterations = ((Number) params.get("max_iterations")).intValue();
        }
        if (params.containsKey("tolerance")) {
            this.tolerance = ((Number) params.get("tolerance")).doubleValue();
        }
    }
}

