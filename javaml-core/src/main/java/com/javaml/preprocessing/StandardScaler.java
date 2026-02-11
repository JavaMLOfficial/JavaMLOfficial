package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.HashMap;
import java.util.Map;

/**
 * Standardizes features by removing the mean and scaling to unit variance.
 * Equivalent to scikit-learn's StandardScaler.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class StandardScaler extends BaseEstimator implements Transformer {
    
    private double[] mean;
    private double[] scale;
    private boolean fitted = false;
    
    /**
     * Fits the scaler to the data.
     * 
     * @param X the input data (2D array: samples x features)
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
        
        mean = new double[nFeatures];
        scale = new double[nFeatures];
        
        // Calculate mean and standard deviation for each feature
        for (int i = 0; i < nFeatures; i++) {
            double[] featureData = new double[nSamples];
            for (int j = 0; j < nSamples; j++) {
                featureData[j] = X.get(j, i);
            }
            NDArray featureArray = new NDArray(featureData, nSamples);
            mean[i] = ArrayStats.mean(featureArray);
            scale[i] = ArrayStats.std(featureArray);
            
            // Avoid division by zero
            if (scale[i] == 0.0) {
                scale[i] = 1.0;
            }
        }
        
        fitted = true;
    }
    
    /**
     * Transforms the data using the fitted parameters.
     * 
     * @param X the input data
     * @return the standardized data
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
            throw new IllegalArgumentException("Input must be 2D (samples x features)");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nFeatures != mean.length) {
            throw new IllegalArgumentException(
                String.format("Number of features (%d) does not match fitted features (%d)",
                    nFeatures, mean.length));
        }
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = (X.get(i, j) - mean[j]) / scale[j];
                result.set(value, i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Gets the mean values.
     * 
     * @return the mean values
     */
    public double[] getMean() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted first");
        }
        return mean.clone();
    }
    
    /**
     * Gets the scale values (standard deviations).
     * 
     * @return the scale values
     */
    public double[] getScale() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted first");
        }
        return scale.clone();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        // StandardScaler doesn't have configurable parameters
    }
}

