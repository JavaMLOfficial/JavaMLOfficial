package com.javaml.naive_bayes;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * Bernoulli Naive Bayes classifier.
 * Equivalent to scikit-learn's BernoulliNB.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class BernoulliNB extends BaseEstimator implements Estimator {
    
    private double alpha; // Smoothing parameter
    private double binarize; // Threshold for binarization
    private Map<Double, double[]> featureProbs; // class -> feature probabilities
    private Map<Double, Double> classPriors; // class -> prior probability
    private Set<Double> classes;
    private int nFeatures;
    private boolean fitted = false;
    
    /**
     * Creates a new BernoulliNB with default parameters.
     */
    public BernoulliNB() {
        this(1.0, 0.0);
    }
    
    /**
     * Creates a new BernoulliNB with specified parameters.
     * 
     * @param alpha smoothing parameter
     * @param binarize threshold for binarization (0.0 = no binarization)
     */
    public BernoulliNB(double alpha, double binarize) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
        this.binarize = binarize;
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
        
        // Binarize if needed
        NDArray XBinarized = binarize(X);
        
        // Get unique classes
        classes = new HashSet<>();
        for (int i = 0; i < nSamples; i++) {
            classes.add(y.get(i));
        }
        
        featureProbs = new HashMap<>();
        classPriors = new HashMap<>();
        
        // For each class, compute feature probabilities
        for (Double cls : classes) {
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                if (y.get(i) == cls) {
                    classIndices.add(i);
                }
            }
            
            int nClassSamples = classIndices.size();
            classPriors.put(cls, (double) nClassSamples / nSamples);
            
            double[] probs = new double[nFeatures];
            
            for (int feature = 0; feature < nFeatures; feature++) {
                int count = 0;
                for (int idx : classIndices) {
                    if (XBinarized.get(idx, feature) > 0.5) {
                        count++;
                    }
                }
                // Laplace smoothing
                probs[feature] = (count + alpha) / (nClassSamples + 2 * alpha);
            }
            
            featureProbs.put(cls, probs);
        }
        
        fitted = true;
    }
    
    /**
     * Binarizes the input data.
     */
    private NDArray binarize(NDArray X) {
        if (binarize == 0.0) {
            return new NDArray(X);
        }
        
        int[] shape = X.getShape();
        NDArray result = new NDArray(shape[0], shape[1]);
        
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                result.set(X.get(i, j) > binarize ? 1.0 : 0.0, i, j);
            }
        }
        
        return result;
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
        
        NDArray XBinarized = binarize(X);
        int nSamples = XBinarized.getShape()[0];
        double[] predictions = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            predictions[i] = predictSingle(XBinarized, i);
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
            
            double[] probs = featureProbs.get(cls);
            
            for (int feature = 0; feature < nFeatures; feature++) {
                double value = X.get(sampleIndex, feature);
                double prob = probs[feature];
                
                if (value > 0.5) {
                    logProb += Math.log(prob);
                } else {
                    logProb += Math.log(1.0 - prob);
                }
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
        params.put("binarize", binarize);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) {
            this.alpha = ((Number) params.get("alpha")).doubleValue();
        }
        if (params.containsKey("binarize")) {
            this.binarize = ((Number) params.get("binarize")).doubleValue();
        }
    }
}

