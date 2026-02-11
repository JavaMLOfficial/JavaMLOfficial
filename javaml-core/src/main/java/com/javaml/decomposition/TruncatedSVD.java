package com.javaml.decomposition;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Truncated Singular Value Decomposition (SVD).
 * Equivalent to scikit-learn's TruncatedSVD.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class TruncatedSVD extends BaseEstimator implements Transformer {
    
    private int nComponents;
    private NDArray components;
    private double[] explainedVariance;
    private double[] singularValues;
    private boolean fitted = false;
    
    /**
     * Creates a new TruncatedSVD with default 2 components.
     */
    public TruncatedSVD() {
        this(2);
    }
    
    /**
     * Creates a new TruncatedSVD with specified number of components.
     * 
     * @param nComponents number of components to keep
     */
    public TruncatedSVD(int nComponents) {
        if (nComponents <= 0) {
            throw new IllegalArgumentException("nComponents must be positive");
        }
        this.nComponents = nComponents;
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
        int nFeatures = shape[1];
        
        int actualNComponents = Math.min(nComponents, Math.min(nSamples, nFeatures));
        
        // Simplified SVD using power iteration
        // For production, use proper SVD algorithm (e.g., Lanczos)
        components = computeTruncatedSVD(X, actualNComponents);
        
        // Compute explained variance
        explainedVariance = computeExplainedVariance(X, components);
        
        fitted = true;
    }
    
    /**
     * Computes truncated SVD using power iteration (simplified).
     */
    private NDArray computeTruncatedSVD(NDArray X, int nComponents) {
        int nSamples = X.getShape()[0];
        int nFeatures = X.getShape()[1];
        
        // Compute X^T * X for right singular vectors
        NDArray XT = X.transpose();
        NDArray XTX = new NDArray(nFeatures, nFeatures);
        
        for (int i = 0; i < nFeatures; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0.0;
                for (int k = 0; k < nSamples; k++) {
                    sum += XT.get(i, k) * X.get(k, j);
                }
                XTX.set(sum, i, j);
            }
        }
        
        // Compute eigenvectors of X^T * X (right singular vectors)
        NDArray components = new NDArray(nComponents, nFeatures);
        singularValues = new double[nComponents];
        
        for (int comp = 0; comp < nComponents; comp++) {
            // Power iteration for eigenvector
            double[] vector = new double[nFeatures];
            java.util.Random random = new java.util.Random();
            for (int i = 0; i < nFeatures; i++) {
                vector[i] = random.nextGaussian();
            }
            
            // Normalize
            double norm = 0.0;
            for (double v : vector) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);
            for (int i = 0; i < nFeatures; i++) {
                vector[i] /= norm;
            }
            
            // Power iteration
            for (int iter = 0; iter < 100; iter++) {
                double[] newVector = new double[nFeatures];
                for (int i = 0; i < nFeatures; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < nFeatures; j++) {
                        sum += XTX.get(i, j) * vector[j];
                    }
                    newVector[i] = sum;
                }
                
                // Normalize
                norm = 0.0;
                for (double v : newVector) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);
                for (int i = 0; i < nFeatures; i++) {
                    newVector[i] /= norm;
                }
                
                vector = newVector;
            }
            
            // Store component
            for (int i = 0; i < nFeatures; i++) {
                components.set(vector[i], comp, i);
            }
            
            // Compute singular value
            double sv = 0.0;
            for (int i = 0; i < nFeatures; i++) {
                double sum = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    sum += XTX.get(i, j) * vector[j];
                }
                sv += sum * vector[i];
            }
            singularValues[comp] = Math.sqrt(Math.max(0, sv));
        }
        
        return components;
    }
    
    /**
     * Computes explained variance for each component.
     */
    private double[] computeExplainedVariance(NDArray X, NDArray components) {
        int nComponents = components.getShape()[0];
        double[] variance = new double[nComponents];
        
        for (int i = 0; i < nComponents; i++) {
            variance[i] = singularValues[i] * singularValues[i];
        }
        
        return variance;
    }
    
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("TruncatedSVD must be fitted before transform");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D");
        }
        
        // Project onto components: X * V^T
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nComponents = components.getShape()[0];
        
        NDArray result = new NDArray(nSamples, nComponents);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nComponents; j++) {
                double sum = 0.0;
                int nFeatures = X.getShape()[1];
                for (int k = 0; k < nFeatures; k++) {
                    sum += X.get(i, k) * components.get(j, k);
                }
                result.set(sum, i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Gets the components (right singular vectors).
     * 
     * @return the components
     */
    public NDArray getComponents() {
        if (!fitted) {
            throw new IllegalStateException("TruncatedSVD must be fitted first");
        }
        return new NDArray(components);
    }
    
    /**
     * Gets the singular values.
     * 
     * @return the singular values
     */
    public double[] getSingularValues() {
        if (!fitted) {
            throw new IllegalStateException("TruncatedSVD must be fitted first");
        }
        return singularValues.clone();
    }
    
    /**
     * Gets the explained variance for each component.
     * 
     * @return the explained variance
     */
    public double[] getExplainedVariance() {
        if (!fitted) {
            throw new IllegalStateException("TruncatedSVD must be fitted first");
        }
        return explainedVariance.clone();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("n_components", nComponents);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_components")) {
            this.nComponents = ((Number) params.get("n_components")).intValue();
        }
    }
}

