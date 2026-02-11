package com.javaml.tree;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;
import com.javaml.random.RandomGenerator;

import java.util.*;

/**
 * Random Forest Regressor.
 * Equivalent to scikit-learn's RandomForestRegressor.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class RandomForestRegressor extends BaseEstimator implements Estimator {
    
    private List<DecisionTreeRegressor> trees;
    private int nEstimators;
    private int maxFeatures;
    private RandomGenerator random;
    private boolean fitted = false;
    
    /**
     * Creates a new RandomForestRegressor with default parameters.
     */
    public RandomForestRegressor() {
        this(100, -1);
    }
    
    /**
     * Creates a new RandomForestRegressor with specified parameters.
     * 
     * @param nEstimators number of trees in the forest
     * @param maxFeatures number of features to consider for best split (-1 for sqrt)
     */
    public RandomForestRegressor(int nEstimators, int maxFeatures) {
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
            DecisionTreeRegressor tree = new DecisionTreeRegressor();
            tree.fit(X_subset, y_subset);
            
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
        
        // For each sample, average predictions from all trees
        for (int i = 0; i < nSamples; i++) {
            double sum = 0.0;
            
            for (DecisionTreeRegressor tree : trees) {
                // Extract single sample
                int nFeatures = X.getShape()[1];
                NDArray singleSample = new NDArray(1, nFeatures);
                for (int j = 0; j < nFeatures; j++) {
                    singleSample.set(X.get(i, j), 0, j);
                }
                NDArray treePred = tree.predict(singleSample);
                sum += treePred.get(0);
            }
            
            predictions[i] = sum / trees.size();
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

