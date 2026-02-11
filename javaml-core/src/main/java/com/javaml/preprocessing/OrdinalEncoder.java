package com.javaml.preprocessing;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Encodes categorical features as ordinal integers.
 * Equivalent to scikit-learn's OrdinalEncoder.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class OrdinalEncoder extends BaseEstimator implements Transformer {
    
    private List<Map<Double, Integer>> categories;
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
        
        categories = new ArrayList<>();
        
        // For each feature, collect unique categories and assign ordinal values
        for (int feature = 0; feature < nFeatures; feature++) {
            Set<Double> uniqueValues = new LinkedHashSet<>();
            for (int sample = 0; sample < nSamples; sample++) {
                uniqueValues.add(X.get(sample, feature));
            }
            
            Map<Double, Integer> categoryMap = new LinkedHashMap<>();
            int ordinal = 0;
            for (Double value : uniqueValues) {
                categoryMap.put(value, ordinal++);
            }
            
            categories.add(categoryMap);
        }
        
        fitted = true;
    }
    
    /**
     * Transforms the data using the fitted encoder.
     * 
     * @param X the input data
     * @return the encoded data
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
        
        if (nFeatures != categories.size()) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted encoder");
        }
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = X.get(i, j);
                Map<Double, Integer> categoryMap = categories.get(j);
                
                if (categoryMap.containsKey(value)) {
                    result.set(categoryMap.get(value), i, j);
                } else {
                    // Unknown category - use -1 or throw exception
                    throw new IllegalArgumentException(
                        "Unknown category " + value + " for feature " + j);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Gets the categories for each feature.
     * 
     * @return list of category maps
     */
    public List<Map<Double, Integer>> getCategories() {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted first");
        }
        return new ArrayList<>(categories);
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        // OrdinalEncoder doesn't have configurable parameters
    }
}

