package com.javaml.naive_bayes;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * Multinomial Naive Bayes classifier.
 * Equivalent to scikit-learn's MultinomialNB.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class MultinomialNB extends BaseEstimator implements Estimator {
    
    private double alpha; // Smoothing parameter
    private Map<Double, double[]> featureCounts; // class -> feature counts
    private Map<Double, Double> classPriors; // class -> prior probability
    private Map<Double, Double> classTotals; // class -> total count
    private Set<Double> classes;
    private int nFeatures;
    private boolean fitted = false;
    
    /**
     * Creates a new MultinomialNB with default alpha=1.0.
     */
    public MultinomialNB() {
        this(1.0);
    }
    
    /**
     * Creates a new MultinomialNB with specified alpha.
     * 
     * @param alpha smoothing parameter
     */
    public MultinomialNB(double alpha) {
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
        nFeatures = shape[1];
        
        if (nSamples != y.getSize()) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        // Get unique classes
        classes = new HashSet<>();
        for (int i = 0; i < nSamples; i++) {
            classes.add(y.get(i));
        }
        
        featureCounts = new HashMap<>();
        classPriors = new HashMap<>();
        classTotals = new HashMap<>();
        
        // For each class, compute feature counts
        for (Double cls : classes) {
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                if (y.get(i) == cls) {
                    classIndices.add(i);
                }
            }
            
            int nClassSamples = classIndices.size();
            classPriors.put(cls, (double) nClassSamples / nSamples);
            
            double[] counts = new double[nFeatures];
            double total = 0.0;
            
            for (int feature = 0; feature < nFeatures; feature++) {
                double sum = 0.0;
                for (int idx : classIndices) {
                    double value = X.get(idx, feature);
                    if (value < 0) {
                        throw new IllegalArgumentException(
                            "MultinomialNB requires non-negative feature values");
                    }
                    sum += value;
                }
                counts[feature] = sum;
                total += sum;
            }
            
            featureCounts.put(cls, counts);
            classTotals.put(cls, total);
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
            
            double[] counts = featureCounts.get(cls);
            double total = classTotals.get(cls);
            
            for (int feature = 0; feature < nFeatures; feature++) {
                double value = X.get(sampleIndex, feature);
                double count = counts[feature];
                
                // Multinomial log probability with smoothing
                double prob = (count + alpha) / (total + alpha * nFeatures);
                logProb += value * Math.log(prob);
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

