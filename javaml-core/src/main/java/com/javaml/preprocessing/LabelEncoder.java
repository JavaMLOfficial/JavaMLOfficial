package com.javaml.preprocessing;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Encodes labels with values between 0 and n_classes-1.
 * Equivalent to scikit-learn's LabelEncoder.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class LabelEncoder extends BaseEstimator implements Transformer {
    
    private Map<Object, Integer> labelToIndex;
    private Map<Integer, Object> indexToLabel;
    private boolean fitted = false;
    
    /**
     * Fits the encoder to the labels.
     * 
     * @param y the labels (1D array)
     */
    @Override
    public void fit(NDArray y) {
        if (y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (y.getNdims() != 1) {
            throw new IllegalArgumentException("Input must be 1D");
        }
        
        labelToIndex = new LinkedHashMap<>();
        indexToLabel = new HashMap<>();
        
        Set<Object> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < y.getSize(); i++) {
            uniqueLabels.add(y.get(i));
        }
        
        int index = 0;
        for (Object label : uniqueLabels) {
            labelToIndex.put(label, index);
            indexToLabel.put(index, label);
            index++;
        }
        
        fitted = true;
    }
    
    /**
     * Transforms labels to encoded values.
     * 
     * @param y the labels
     * @return the encoded labels
     */
    @Override
    public NDArray transform(NDArray y) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before transform");
        }
        if (y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (y.getNdims() != 1) {
            throw new IllegalArgumentException("Input must be 1D");
        }
        
        double[] encoded = new double[y.getSize()];
        for (int i = 0; i < y.getSize(); i++) {
            Object label = y.get(i);
            if (!labelToIndex.containsKey(label)) {
                throw new IllegalArgumentException("Unknown label: " + label);
            }
            encoded[i] = labelToIndex.get(label);
        }
        
        return new NDArray(encoded, encoded.length);
    }
    
    /**
     * Inverse transforms encoded values back to labels.
     * 
     * @param y the encoded labels
     * @return the original labels
     */
    public NDArray inverseTransform(NDArray y) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before inverse transform");
        }
        if (y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (y.getNdims() != 1) {
            throw new IllegalArgumentException("Input must be 1D");
        }
        
        double[] decoded = new double[y.getSize()];
        for (int i = 0; i < y.getSize(); i++) {
            int index = (int) y.get(i);
            if (!indexToLabel.containsKey(index)) {
                throw new IllegalArgumentException("Unknown index: " + index);
            }
            Object label = indexToLabel.get(index);
            decoded[i] = label instanceof Number ? 
                ((Number) label).doubleValue() : Double.NaN;
        }
        
        return new NDArray(decoded, decoded.length);
    }
    
    /**
     * Gets the number of classes.
     * 
     * @return the number of classes
     */
    public int getNClasses() {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted first");
        }
        return labelToIndex.size();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("fitted", fitted);
        params.put("n_classes", fitted ? labelToIndex.size() : 0);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        // LabelEncoder doesn't have configurable parameters
    }
}

