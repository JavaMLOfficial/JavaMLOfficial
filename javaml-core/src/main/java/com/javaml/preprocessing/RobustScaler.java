package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.HashMap;
import java.util.Map;

/**
 * Scales features using statistics that are robust to outliers.
 * Equivalent to scikit-learn's RobustScaler.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class RobustScaler extends BaseEstimator implements Transformer {
    
    private double[] center; // median
    private double[] scale; // IQR (interquartile range)
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
        
        center = new double[nFeatures];
        scale = new double[nFeatures];
        
        // Calculate median and IQR for each feature
        for (int i = 0; i < nFeatures; i++) {
            double[] featureData = new double[nSamples];
            for (int j = 0; j < nSamples; j++) {
                featureData[j] = X.get(j, i);
            }
            NDArray featureArray = new NDArray(featureData, nSamples);
            
            center[i] = ArrayStats.median(featureArray);
            
            // Calculate IQR (Q3 - Q1)
            double q1 = ArrayStats.percentile(featureArray, 25.0);
            double q3 = ArrayStats.percentile(featureArray, 75.0);
            scale[i] = q3 - q1;
            
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
        
        if (nFeatures != center.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted scaler");
        }
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = (X.get(i, j) - center[j]) / scale[j];
                result.set(value, i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Gets the center values (medians).
     * 
     * @return the center values
     */
    public double[] getCenter() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted first");
        }
        return center.clone();
    }
    
    /**
     * Gets the scale values (IQR).
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
        // RobustScaler doesn't have configurable parameters
    }
}

