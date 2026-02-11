package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.HashMap;
import java.util.Map;

/**
 * Transforms features by scaling each feature to a given range (default: [0, 1]).
 * Equivalent to scikit-learn's MinMaxScaler.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class MinMaxScaler extends BaseEstimator implements Transformer {
    
    private double min = 0.0;
    private double max = 1.0;
    private double[] dataMin;
    private double[] dataMax;
    private double[] scale;
    private double[] min_;
    private boolean fitted = false;
    
    /**
     * Creates a new MinMaxScaler with default range [0, 1].
     */
    public MinMaxScaler() {
        this(0.0, 1.0);
    }
    
    /**
     * Creates a new MinMaxScaler with the specified range.
     * 
     * @param min the minimum value of the transformed data
     * @param max the maximum value of the transformed data
     */
    public MinMaxScaler(double min, double max) {
        if (min >= max) {
            throw new IllegalArgumentException("min must be less than max");
        }
        this.min = min;
        this.max = max;
    }
    
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
        
        dataMin = new double[nFeatures];
        dataMax = new double[nFeatures];
        scale = new double[nFeatures];
        min_ = new double[nFeatures];
        
        // Calculate min and max for each feature
        for (int i = 0; i < nFeatures; i++) {
            double[] featureData = new double[nSamples];
            for (int j = 0; j < nSamples; j++) {
                featureData[j] = X.get(j, i);
            }
            NDArray featureArray = new NDArray(featureData, nSamples);
            dataMin[i] = ArrayStats.min(featureArray);
            dataMax[i] = ArrayStats.max(featureArray);
            
            // Calculate scale
            double dataRange = dataMax[i] - dataMin[i];
            if (dataRange == 0.0) {
                scale[i] = 1.0;
                min_[i] = min;
            } else {
                scale[i] = (max - min) / dataRange;
                min_[i] = min - dataMin[i] * scale[i];
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
            throw new IllegalArgumentException("Input must be 2D (samples x features)");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nFeatures != dataMin.length) {
            throw new IllegalArgumentException(
                String.format("Number of features (%d) does not match fitted features (%d)",
                    nFeatures, dataMin.length));
        }
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = X.get(i, j) * scale[j] + min_[j];
                result.set(value, i, j);
            }
        }
        
        return result;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("min", min);
        params.put("max", max);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("min")) {
            this.min = ((Number) params.get("min")).doubleValue();
        }
        if (params.containsKey("max")) {
            this.max = ((Number) params.get("max")).doubleValue();
        }
    }
}

