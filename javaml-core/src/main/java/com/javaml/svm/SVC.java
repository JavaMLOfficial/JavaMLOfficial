package com.javaml.svm;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * Support Vector Classifier (SVC).
 * Equivalent to scikit-learn's SVC.
 * 
 * Note: This is a simplified implementation using gradient descent.
 * For production use, consider using a more sophisticated optimization algorithm.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class SVC extends BaseEstimator implements Estimator {
    
    private double C;
    private String kernel;
    private double gamma;
    private int maxIterations;
    private double[] supportVectors;
    private double[] coefficients;
    private double intercept;
    private Set<Double> classes;
    private NDArray XTrain; // Store training data for prediction
    private double[] yBinaryTrain; // Store binary labels
    private boolean fitted = false;
    
    /**
     * Creates a new SVC with default parameters.
     */
    public SVC() {
        this(1.0, "rbf", 1.0, 1000);
    }
    
    /**
     * Creates a new SVC with specified parameters.
     * 
     * @param C regularization parameter
     * @param kernel kernel type ("linear" or "rbf")
     * @param gamma kernel coefficient for RBF
     * @param maxIterations maximum number of iterations
     */
    public SVC(double C, String kernel, double gamma, int maxIterations) {
        if (C <= 0) {
            throw new IllegalArgumentException("C must be positive");
        }
        this.C = C;
        this.kernel = kernel;
        this.gamma = gamma;
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
        
        // Get unique classes (binary classification for now)
        classes = new HashSet<>();
        for (int i = 0; i < nSamples; i++) {
            classes.add(y.get(i));
        }
        
        if (classes.size() != 2) {
            throw new IllegalArgumentException("SVC currently supports binary classification only");
        }
        
        List<Double> classList = new ArrayList<>(classes);
        double class0 = classList.get(0);
        double class1 = classList.get(1);
        
        // Convert to -1, +1 labels
        double[] yBinary = new double[nSamples];
        for (int i = 0; i < nSamples; i++) {
            yBinary[i] = y.get(i) == class0 ? -1.0 : 1.0;
        }
        
        // Store training data for prediction
        this.XTrain = new NDArray(X);
        this.yBinaryTrain = yBinary;
        
        // Simplified SVC training using gradient descent
        // For production, use SMO or other specialized algorithms
        trainSVM(X, yBinary);
        
        fitted = true;
    }
    
    /**
     * Trains SVM using simplified gradient descent.
     */
    private void trainSVM(NDArray X, double[] yBinary) {
        int nSamples = X.getShape()[0];
        int nFeatures = X.getShape()[1];
        
        // Initialize weights
        coefficients = new double[nSamples];
        intercept = 0.0;
        double learningRate = 0.01;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            for (int i = 0; i < nSamples; i++) {
                // Compute prediction
                double prediction = intercept;
                for (int j = 0; j < nSamples; j++) {
                    double kernelValue = computeKernel(X, i, j);
                    prediction += coefficients[j] * yBinary[j] * kernelValue;
                }
                
                // Hinge loss gradient
                double margin = yBinary[i] * prediction;
                if (margin < 1.0) {
                    // Update coefficient
                    double gradient = -yBinary[i] * computeKernel(X, i, i) + C * coefficients[i];
                    coefficients[i] -= learningRate * gradient;
                    coefficients[i] = Math.max(0, Math.min(C, coefficients[i])); // Clip to [0, C]
                    
                    // Update intercept
                    intercept += learningRate * yBinary[i];
                }
            }
        }
        
        // Store support vectors (non-zero coefficients)
        List<Integer> supportIndices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            if (Math.abs(coefficients[i]) > 1e-6) {
                supportIndices.add(i);
            }
        }
        
        supportVectors = new double[supportIndices.size()];
        for (int i = 0; i < supportIndices.size(); i++) {
            supportVectors[i] = supportIndices.get(i);
        }
    }
    
    /**
     * Computes kernel value between two samples.
     */
    private double computeKernel(NDArray X, int i, int j) {
        int nFeatures = X.getShape()[1];
        
        if ("linear".equals(kernel)) {
            double dot = 0.0;
            for (int k = 0; k < nFeatures; k++) {
                dot += X.get(i, k) * X.get(j, k);
            }
            return dot;
        } else if ("rbf".equals(kernel)) {
            double distance = 0.0;
            for (int k = 0; k < nFeatures; k++) {
                double diff = X.get(i, k) - X.get(j, k);
                distance += diff * diff;
            }
            return Math.exp(-gamma * distance);
        } else {
            throw new UnsupportedOperationException("Kernel " + kernel + " not yet implemented");
        }
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
        
        int nSamples = X.getShape()[0];
        double[] predictions = new double[nSamples];
        
        List<Double> classList = new ArrayList<>(classes);
        double class0 = classList.get(0);
        double class1 = classList.get(1);
        
        // Use stored training data
        
        for (int i = 0; i < nSamples; i++) {
            double decision = intercept;
            for (int j = 0; j < supportVectors.length; j++) {
                int svIdx = (int) supportVectors[j];
                double kernelValue = computeKernelBetween(X, i, XTrain, svIdx);
                decision += coefficients[svIdx] * yBinaryTrain[svIdx] * kernelValue;
            }
            
            predictions[i] = decision >= 0 ? class1 : class0;
        }
        
        return new NDArray(predictions, predictions.length);
    }
    
    /**
     * Computes kernel between a test sample and training sample.
     */
    private double computeKernelBetween(NDArray XTest, int testIdx, NDArray XTrain, int trainIdx) {
        int nFeatures = XTest.getShape()[1];
        
        if ("linear".equals(kernel)) {
            double dot = 0.0;
            for (int k = 0; k < nFeatures; k++) {
                dot += XTest.get(testIdx, k) * XTrain.get(trainIdx, k);
            }
            return dot;
        } else if ("rbf".equals(kernel)) {
            double distance = 0.0;
            for (int k = 0; k < nFeatures; k++) {
                double diff = XTest.get(testIdx, k) - XTrain.get(trainIdx, k);
                distance += diff * diff;
            }
            return Math.exp(-gamma * distance);
        } else {
            throw new UnsupportedOperationException("Kernel " + kernel + " not yet implemented");
        }
    }
    
    
    @Override
    public double score(NDArray X, NDArray y) {
        NDArray predictions = predict(X);
        return calculateAccuracy(predictions, y);
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
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("C", C);
        params.put("kernel", kernel);
        params.put("gamma", gamma);
        params.put("max_iterations", maxIterations);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("C")) {
            this.C = ((Number) params.get("C")).doubleValue();
        }
        if (params.containsKey("kernel")) {
            this.kernel = (String) params.get("kernel");
        }
        if (params.containsKey("gamma")) {
            this.gamma = ((Number) params.get("gamma")).doubleValue();
        }
        if (params.containsKey("max_iterations")) {
            this.maxIterations = ((Number) params.get("max_iterations")).intValue();
        }
    }
}

