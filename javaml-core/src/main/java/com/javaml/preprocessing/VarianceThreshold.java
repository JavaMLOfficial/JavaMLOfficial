package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Feature selector that removes low-variance features.
 * Equivalent to scikit-learn's VarianceThreshold.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class VarianceThreshold extends BaseEstimator implements Transformer {
    
    private double threshold;
    private boolean[] featureMask; // true = keep, false = remove
    private int nFeatures;
    private boolean fitted = false;
    
    /**
     * Creates a new VarianceThreshold with default threshold=0.0.
     */
    public VarianceThreshold() {
        this(0.0);
    }
    
    /**
     * Creates a new VarianceThreshold with specified threshold.
     * 
     * @param threshold features with variance below this threshold will be removed
     */
    public VarianceThreshold(double threshold) {
        if (threshold < 0) {
            throw new IllegalArgumentException("Threshold must be non-negative");
        }
        this.threshold = threshold;
    }
    
    @Override
    public void fit(NDArray X) {
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D (samples x features)");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        nFeatures = shape[1];
        
        featureMask = new boolean[nFeatures];
        
        // Compute variance for each feature
        for (int feature = 0; feature < nFeatures; feature++) {
            double[] featureData = new double[nSamples];
            for (int sample = 0; sample < nSamples; sample++) {
                featureData[sample] = X.get(sample, feature);
            }
            NDArray featureArray = new NDArray(featureData, nSamples);
            double variance = ArrayStats.var(featureArray);
            
            featureMask[feature] = variance >= threshold;
        }
        
        fitted = true;
    }
    
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("VarianceThreshold must be fitted before transform");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        
        if (shape[1] != nFeatures) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted transformer");
        }
        
        // Count features to keep
        int nFeaturesToKeep = 0;
        for (boolean keep : featureMask) {
            if (keep) nFeaturesToKeep++;
        }
        
        // Create result array with only selected features
        NDArray result = new NDArray(nSamples, nFeaturesToKeep);
        
        int resultCol = 0;
        for (int feature = 0; feature < nFeatures; feature++) {
            if (featureMask[feature]) {
                for (int sample = 0; sample < nSamples; sample++) {
                    result.set(X.get(sample, feature), sample, resultCol);
                }
                resultCol++;
            }
        }
        
        return result;
    }
    
    /**
     * Gets the feature mask (which features are kept).
     * 
     * @return boolean array where true means feature is kept
     */
    public boolean[] getFeatureMask() {
        if (!fitted) {
            throw new IllegalStateException("VarianceThreshold must be fitted first");
        }
        return featureMask.clone();
    }
    
    /**
     * Gets the number of features that will be kept.
     * 
     * @return number of features to keep
     */
    public int getNFeaturesToKeep() {
        if (!fitted) {
            throw new IllegalStateException("VarianceThreshold must be fitted first");
        }
        int count = 0;
        for (boolean keep : featureMask) {
            if (keep) count++;
        }
        return count;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("threshold", threshold);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("threshold")) {
            this.threshold = ((Number) params.get("threshold")).doubleValue();
        }
    }
}

