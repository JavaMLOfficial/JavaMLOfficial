package com.javaml.preprocessing;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * One-hot encoding transformer.
 * Equivalent to scikit-learn's OneHotEncoder.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class OneHotEncoder extends BaseEstimator implements Transformer {
    
    private Map<Integer, Map<Double, Integer>> categories;
    private int[] nCategories;
    private boolean fitted = false;
    
    /**
     * Fits the encoder to the data.
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
        
        categories = new HashMap<>();
        nCategories = new int[nFeatures];
        
        // Find unique categories for each feature
        for (int feature = 0; feature < nFeatures; feature++) {
            Set<Double> uniqueValues = new LinkedHashSet<>();
            for (int sample = 0; sample < nSamples; sample++) {
                uniqueValues.add(X.get(sample, feature));
            }
            
            Map<Double, Integer> categoryMap = new LinkedHashMap<>();
            int index = 0;
            for (Double value : uniqueValues) {
                categoryMap.put(value, index++);
            }
            
            categories.put(feature, categoryMap);
            nCategories[feature] = categoryMap.size();
        }
        
        fitted = true;
    }
    
    /**
     * Transforms the data to one-hot encoding.
     * 
     * @param X the input data
     * @return the one-hot encoded data
     */
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before transform");
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
        
        if (nFeatures != nCategories.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted encoder");
        }
        
        // Calculate output size
        int outputSize = 0;
        for (int nCat : nCategories) {
            outputSize += nCat;
        }
        
        NDArray result = new NDArray(nSamples, outputSize);
        
        int outputCol = 0;
        for (int feature = 0; feature < nFeatures; feature++) {
            Map<Double, Integer> categoryMap = categories.get(feature);
            
            for (int sample = 0; sample < nSamples; sample++) {
                double value = X.get(sample, feature);
                Integer categoryIndex = categoryMap.get(value);
                
                if (categoryIndex == null) {
                    throw new IllegalArgumentException(
                        "Unknown category " + value + " for feature " + feature);
                }
                
                // Set the corresponding one-hot position to 1
                result.set(1.0, sample, outputCol + categoryIndex);
            }
            
            outputCol += nCategories[feature];
        }
        
        return result;
    }
    
    /**
     * Gets the number of categories for each feature.
     * 
     * @return array of category counts
     */
    public int[] getNCategories() {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted first");
        }
        return nCategories.clone();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("fitted", fitted);
        if (fitted) {
            params.put("n_features", nCategories.length);
            params.put("n_categories", Arrays.toString(nCategories));
        }
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        // OneHotEncoder doesn't have configurable parameters
    }
}

