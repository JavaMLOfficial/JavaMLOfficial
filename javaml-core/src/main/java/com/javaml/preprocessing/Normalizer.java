package com.javaml.preprocessing;

import com.javaml.array.NDArray;
import com.javaml.array.LinearAlgebra;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.HashMap;
import java.util.Map;

/**
 * Normalizes samples individually to unit norm.
 * Equivalent to scikit-learn's Normalizer.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class Normalizer extends BaseEstimator implements Transformer {
    
    public enum Norm {
        L1, L2, MAX
    }
    
    private Norm norm;
    private boolean fitted = false;
    
    /**
     * Creates a new Normalizer with L2 norm (default).
     */
    public Normalizer() {
        this(Norm.L2);
    }
    
    /**
     * Creates a new Normalizer with specified norm.
     * 
     * @param norm the norm to use (L1, L2, or MAX)
     */
    public Normalizer(Norm norm) {
        this.norm = norm;
    }
    
    @Override
    public void fit(NDArray X) {
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D (samples x features)");
        }
        // Normalizer doesn't need to fit - it's stateless
        fitted = true;
    }
    
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Normalizer must be fitted before transform");
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
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            // Calculate norm for this sample
            double sampleNorm = calculateNorm(X, i, norm);
            
            if (sampleNorm > 0) {
                for (int j = 0; j < nFeatures; j++) {
                    result.set(X.get(i, j) / sampleNorm, i, j);
                }
            } else {
                // Zero norm - keep as is
                for (int j = 0; j < nFeatures; j++) {
                    result.set(X.get(i, j), i, j);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Calculates the norm for a sample.
     */
    private double calculateNorm(NDArray X, int sampleIndex, Norm normType) {
        int nFeatures = X.getShape()[1];
        
        switch (normType) {
            case L1:
                double l1Sum = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    l1Sum += Math.abs(X.get(sampleIndex, j));
                }
                return l1Sum;
                
            case L2:
                double l2Sum = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    double val = X.get(sampleIndex, j);
                    l2Sum += val * val;
                }
                return Math.sqrt(l2Sum);
                
            case MAX:
                double maxVal = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    maxVal = Math.max(maxVal, Math.abs(X.get(sampleIndex, j)));
                }
                return maxVal;
                
            default:
                throw new IllegalArgumentException("Unknown norm type: " + normType);
        }
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("norm", norm);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("norm")) {
            this.norm = (Norm) params.get("norm");
        }
    }
}

