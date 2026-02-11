package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.HashMap;
import java.util.Map;

/**
 * Scales each feature by its maximum absolute value.
 * Equivalent to scikit-learn's MaxAbsScaler.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class MaxAbsScaler extends BaseEstimator implements Transformer {
    
    private double[] maxAbs;
    private boolean fitted = false;
    
    /**
     * Fits the scaler to the data.
     * 
     * @param X the input data (2D: samples x features)
     */
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
        int nFeatures = shape[1];
        
        maxAbs = new double[nFeatures];
        
        // Calculate maximum absolute value for each feature
        for (int i = 0; i < nFeatures; i++) {
            double max = 0.0;
            for (int j = 0; j < nSamples; j++) {
                double absValue = Math.abs(X.get(j, i));
                max = Math.max(max, absValue);
            }
            maxAbs[i] = max > 0 ? max : 1.0; // Avoid division by zero
        }
        
        fitted = true;
    }
    
    /**
     * Transforms the data using the fitted parameters.
     * 
     * @param X the input data
     * @return the scaled data
     */
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nFeatures != maxAbs.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted scaler");
        }
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = X.get(i, j) / maxAbs[j];
                result.set(value, i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Gets the maximum absolute values.
     * 
     * @return the maximum absolute values
     */
    public double[] getMaxAbs() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted first");
        }
        return maxAbs.clone();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        // MaxAbsScaler doesn't have configurable parameters
    }
}

