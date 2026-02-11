package com.javaml.tree;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;
import com.javaml.random.RandomGenerator;

import java.util.*;

/**
 * Random Forest Classifier.
 * Equivalent to scikit-learn's RandomForestClassifier.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class RandomForestClassifier extends BaseEstimator implements Estimator {
    
    private List<DecisionTreeClassifier> trees;
    private int nEstimators;
    private int maxFeatures;
    private RandomGenerator random;
    private boolean fitted = false;
    
    /**
     * Creates a new RandomForestClassifier with default parameters.
     */
    public RandomForestClassifier() {
        this(100, -1);
    }
    
    /**
     * Creates a new RandomForestClassifier with specified parameters.
     * 
     * @param nEstimators number of trees in the forest
     * @param maxFeatures number of features to consider for best split (-1 for sqrt)
     */
    public RandomForestClassifier(int nEstimators, int maxFeatures) {
        this.nEstimators = nEstimators;
        this.maxFeatures = maxFeatures;
        this.random = new RandomGenerator();
        this.trees = new ArrayList<>();
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
        
        // Determine max_features
        int actualMaxFeatures = maxFeatures;
        if (maxFeatures == -1) {
            actualMaxFeatures = (int) Math.sqrt(nFeatures);
        } else if (maxFeatures < 0) {
            throw new IllegalArgumentException("max_features must be positive or -1");
        }
        actualMaxFeatures = Math.min(actualMaxFeatures, nFeatures);
        
        trees.clear();
        
        // Train each tree
        for (int i = 0; i < nEstimators; i++) {
            // Bootstrap sampling
            List<Integer> bootstrapIndices = bootstrapSample(nSamples);
            
            // Feature sampling
            List<Integer> featureIndices = sampleFeatures(nFeatures, actualMaxFeatures);
            
            // Create subset of data
            NDArray X_subset = createSubset(X, bootstrapIndices, featureIndices);
            NDArray y_subset = createSubset(y, bootstrapIndices);
            
            // Train tree
            DecisionTreeClassifier tree = new DecisionTreeClassifier();
            tree.fit(X_subset, y_subset);
            
            // Store tree and feature mapping
            trees.add(tree);
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
        
        // For each sample, get predictions from all trees and take majority vote
        for (int i = 0; i < nSamples; i++) {
            Map<Double, Integer> votes = new HashMap<>();
            
            for (DecisionTreeClassifier tree : trees) {
                // Extract single sample
                int nFeatures = X.getShape()[1];
                NDArray singleSample = new NDArray(1, nFeatures);
                for (int j = 0; j < nFeatures; j++) {
                    singleSample.set(X.get(i, j), 0, j);
                }
                NDArray treePred = tree.predict(singleSample);
                double pred = treePred.get(0);
                votes.put(pred, votes.getOrDefault(pred, 0) + 1);
            }
            
            // Find majority vote
            double majorityClass = 0.0;
            int maxVotes = 0;
            for (Map.Entry<Double, Integer> entry : votes.entrySet()) {
                if (entry.getValue() > maxVotes) {
                    maxVotes = entry.getValue();
                    majorityClass = entry.getKey();
                }
            }
            
            predictions[i] = majorityClass;
        }
        
        return new NDArray(predictions, predictions.length);
    }
    
    @Override
    public double score(NDArray X, NDArray y) {
        NDArray predictions = predict(X);
        return calculateAccuracy(predictions, y);
    }
    
    /**
     * Creates a bootstrap sample.
     */
    private List<Integer> bootstrapSample(int nSamples) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            int idx = random.randint(0, nSamples).get(0);
            indices.add(idx);
        }
        return indices;
    }
    
    /**
     * Samples features randomly.
     */
    private List<Integer> sampleFeatures(int nFeatures, int maxFeatures) {
        List<Integer> allFeatures = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++) {
            allFeatures.add(i);
        }
        
        Collections.shuffle(allFeatures, new Random());
        return allFeatures.subList(0, maxFeatures);
    }
    
    /**
     * Creates a subset of X with selected samples and features.
     */
    private NDArray createSubset(NDArray X, List<Integer> sampleIndices, List<Integer> featureIndices) {
        int nSamples = sampleIndices.size();
        int nFeatures = featureIndices.size();
        
        NDArray subset = new NDArray(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                subset.set(X.get(sampleIndices.get(i), featureIndices.get(j)), i, j);
            }
        }
        return subset;
    }
    
    /**
     * Creates a subset of y with selected samples.
     */
    private NDArray createSubset(NDArray y, List<Integer> sampleIndices) {
        double[] subset = new double[sampleIndices.size()];
        for (int i = 0; i < sampleIndices.size(); i++) {
            subset[i] = y.get(sampleIndices.get(i));
        }
        return new NDArray(subset, subset.length);
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
        params.put("n_estimators", nEstimators);
        params.put("max_features", maxFeatures);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) {
            this.nEstimators = ((Number) params.get("n_estimators")).intValue();
        }
        if (params.containsKey("max_features")) {
            this.maxFeatures = ((Number) params.get("max_features")).intValue();
        }
    }
}

