package com.javaml.naive_bayes;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * Gaussian Naive Bayes classifier.
 * Equivalent to scikit-learn's GaussianNB.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class GaussianNB extends BaseEstimator implements Estimator {
    
    private Map<Double, double[]> means; // class -> feature means
    private Map<Double, double[]> variances; // class -> feature variances
    private Map<Double, Double> classPriors; // class -> prior probability
    private Set<Double> classes;
    private boolean fitted = false;
    
    /**
     * Creates a new GaussianNB classifier.
     */
    public GaussianNB() {
        this.means = new HashMap<>();
        this.variances = new HashMap<>();
        this.classPriors = new HashMap<>();
        this.classes = new HashSet<>();
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
        
        // Get unique classes
        classes.clear();
        for (int i = 0; i < nSamples; i++) {
            classes.add(y.get(i));
        }
        
        means.clear();
        variances.clear();
        classPriors.clear();
        
        // For each class, compute mean and variance for each feature
        for (Double cls : classes) {
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                if (y.get(i) == cls) {
                    classIndices.add(i);
                }
            }
            
            int nClassSamples = classIndices.size();
            classPriors.put(cls, (double) nClassSamples / nSamples);
            
            double[] classMeans = new double[nFeatures];
            double[] classVariances = new double[nFeatures];
            
            // Compute means
            for (int feature = 0; feature < nFeatures; feature++) {
                double sum = 0.0;
                for (int idx : classIndices) {
                    sum += X.get(idx, feature);
                }
                classMeans[feature] = sum / nClassSamples;
            }
            
            // Compute variances
            for (int feature = 0; feature < nFeatures; feature++) {
                double sum = 0.0;
                for (int idx : classIndices) {
                    double diff = X.get(idx, feature) - classMeans[feature];
                    sum += diff * diff;
                }
                classVariances[feature] = sum / nClassSamples;
                // Add small epsilon to avoid division by zero
                if (classVariances[feature] < 1e-9) {
                    classVariances[feature] = 1e-9;
                }
            }
            
            means.put(cls, classMeans);
            variances.put(cls, classVariances);
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
        
        int nSamples = X.getShape()[0];
        double[] predictions = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            predictions[i] = predictSingle(X, i);
        }
        
        return new NDArray(predictions, predictions.length);
    }
    
    /**
     * Predicts a single sample.
     */
    private double predictSingle(NDArray X, int sampleIndex) {
        double maxLogProb = Double.NEGATIVE_INFINITY;
        double bestClass = 0.0;
        
        for (Double cls : classes) {
            double logProb = Math.log(classPriors.get(cls));
            
            int nFeatures = X.getShape()[1];
            double[] classMeans = means.get(cls);
            double[] classVariances = variances.get(cls);
            
            for (int feature = 0; feature < nFeatures; feature++) {
                double value = X.get(sampleIndex, feature);
                double mean = classMeans[feature];
                double variance = classVariances[feature];
                
                // Gaussian log probability
                double diff = value - mean;
                logProb -= 0.5 * Math.log(2 * Math.PI * variance);
                logProb -= 0.5 * (diff * diff) / variance;
            }
            
            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                bestClass = cls;
            }
        }
        
        return bestClass;
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
    
    /**
     * Gets the class means.
     * 
     * @return map of class to feature means
     */
    public Map<Double, double[]> getMeans() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        Map<Double, double[]> result = new HashMap<>();
        for (Map.Entry<Double, double[]> entry : means.entrySet()) {
            result.put(entry.getKey(), entry.getValue().clone());
        }
        return result;
    }
    
    /**
     * Gets the class variances.
     * 
     * @return map of class to feature variances
     */
    public Map<Double, double[]> getVariances() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        Map<Double, double[]> result = new HashMap<>();
        for (Map.Entry<Double, double[]> entry : variances.entrySet()) {
            result.put(entry.getKey(), entry.getValue().clone());
        }
        return result;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        // GaussianNB doesn't have configurable parameters
    }
}

