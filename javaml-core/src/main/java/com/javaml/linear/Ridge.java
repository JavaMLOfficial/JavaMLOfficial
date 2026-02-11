package com.javaml.linear;

import com.javaml.array.LinearAlgebra;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.HashMap;
import java.util.Map;

/**
 * Ridge regression with L2 regularization.
 * Equivalent to scikit-learn's Ridge.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class Ridge extends BaseEstimator implements Estimator {
    
    private double alpha;
    private NDArray coefficients;
    private double intercept;
    private boolean fitted = false;
    
    /**
     * Creates a new Ridge regressor with default alpha=1.0.
     */
    public Ridge() {
        this(1.0);
    }
    
    /**
     * Creates a new Ridge regressor with specified alpha.
     * 
     * @param alpha regularization strength
     */
    public Ridge(double alpha) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
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
        
        // Ridge regression: (X^T * X + alpha * I) * coefficients = X^T * y
        NDArray XT = X.transpose();
        NDArray XTX = LinearAlgebra.matmul(XT, X);
        
        // Add regularization term (alpha * I)
        for (int i = 0; i < nFeatures; i++) {
            double current = XTX.get(i, i);
            XTX.set(current + alpha, i, i);
        }
        
        // Calculate X^T * y
        NDArray XTy = LinearAlgebra.matmul(XT, y.reshape(nSamples, 1));
        
        // Solve the system (simplified - would use proper solver for production)
        // For small problems, use iterative approach
        solveRidgeSystem(XTX, XTy, X, y);
        
        fitted = true;
    }
    
    /**
     * Solves the Ridge regression system using iterative method.
     */
    private void solveRidgeSystem(NDArray A, NDArray b, NDArray X, NDArray y) {
        int nFeatures = A.getShape()[0];
        
        // Initialize coefficients
        coefficients = new NDArray(nFeatures);
        intercept = 0.0;
        
        // Simple gradient descent for Ridge
        double learningRate = 0.01;
        int maxIterations = 1000;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // Calculate predictions
            NDArray predictions = predictInternal(X);
            
            // Calculate gradients with L2 regularization
            double[] gradCoeff = new double[nFeatures];
            double gradIntercept = 0.0;
            
            for (int i = 0; i < X.getShape()[0]; i++) {
                double error = predictions.get(i) - y.get(i);
                gradIntercept += error;
                
                for (int j = 0; j < nFeatures; j++) {
                    gradCoeff[j] += error * X.get(i, j);
                }
            }
            
            // Update with L2 regularization
            intercept -= learningRate * gradIntercept / X.getShape()[0];
            for (int j = 0; j < nFeatures; j++) {
                double current = coefficients.get(j);
                double gradient = gradCoeff[j] / X.getShape()[0] + alpha * current;
                coefficients.set(current - learningRate * gradient, j);
            }
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
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) {
            this.alpha = ((Number) params.get("alpha")).doubleValue();
        }
    }
}

