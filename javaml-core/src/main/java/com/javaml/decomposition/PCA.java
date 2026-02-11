package com.javaml.decomposition;

import com.javaml.array.ArrayStats;
import com.javaml.array.LinearAlgebra;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Principal Component Analysis (PCA).
 * Equivalent to scikit-learn's PCA.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class PCA extends BaseEstimator implements Transformer {
    
    private int nComponents;
    private NDArray components;
    private double[] explainedVariance;
    private double[] mean;
    private boolean fitted = false;
    
    /**
     * Creates a new PCA with all components (default).
     */
    public PCA() {
        this(-1);
    }
    
    /**
     * Creates a new PCA with specified number of components.
     * 
     * @param nComponents number of components to keep (-1 for all)
     */
    public PCA(int nComponents) {
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
        
        // Center the data (subtract mean)
        mean = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < nSamples; i++) {
                sum += X.get(i, j);
            }
            mean[j] = sum / nSamples;
        }
        
        NDArray XCentered = new NDArray(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                XCentered.set(X.get(i, j) - mean[j], i, j);
            }
        }
        
        // Compute covariance matrix
        NDArray covariance = ArrayStats.cov(XCentered);
        
        // Compute eigenvalues and eigenvectors (simplified - using power iteration)
        // For production, use proper eigenvalue decomposition
        int actualNComponents = nComponents == -1 ? nFeatures : Math.min(nComponents, nFeatures);
        components = computePrincipalComponents(covariance, actualNComponents);
        
        // Compute explained variance
        explainedVariance = computeExplainedVariance(covariance, components);
        
        fitted = true;
    }
    
    /**
     * Computes principal components using power iteration (simplified).
     */
    private NDArray computePrincipalComponents(NDArray covariance, int nComponents) {
        int nFeatures = covariance.getShape()[0];
        NDArray components = new NDArray(nComponents, nFeatures);
        
        // Simplified approach: use first nComponents eigenvectors
        // In production, use proper eigenvalue decomposition (e.g., QR algorithm)
        for (int comp = 0; comp < nComponents; comp++) {
            // Initialize random vector
            double[] vector = new double[nFeatures];
            Random random = new Random();
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
                        sum += covariance.get(i, j) * vector[j];
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
        }
        
        return components;
    }
    
    /**
     * Computes explained variance for each component.
     */
    private double[] computeExplainedVariance(NDArray covariance, NDArray components) {
        int nComponents = components.getShape()[0];
        double[] variance = new double[nComponents];
        
        for (int i = 0; i < nComponents; i++) {
            // Project covariance onto component
            double sum = 0.0;
            int nFeatures = components.getShape()[1];
            for (int j = 0; j < nFeatures; j++) {
                for (int k = 0; k < nFeatures; k++) {
                    sum += components.get(i, j) * covariance.get(j, k) * components.get(i, k);
                }
            }
            variance[i] = sum;
        }
        
        return variance;
    }
    
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("PCA must be fitted before transform");
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
        int nComponents = components.getShape()[0];
        
        if (nFeatures != mean.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted PCA");
        }
        
        // Center the data
        NDArray XCentered = new NDArray(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                XCentered.set(X.get(i, j) - mean[j], i, j);
            }
        }
        
        // Project onto principal components
        NDArray result = LinearAlgebra.matmul(XCentered, components.transpose());
        
        return result;
    }
    
    /**
     * Gets the principal components.
     * 
     * @return the principal components
     */
    public NDArray getComponents() {
        if (!fitted) {
            throw new IllegalStateException("PCA must be fitted first");
        }
        return new NDArray(components);
    }
    
    /**
     * Gets the explained variance for each component.
     * 
     * @return the explained variance
     */
    public double[] getExplainedVariance() {
        if (!fitted) {
            throw new IllegalStateException("PCA must be fitted first");
        }
        return explainedVariance.clone();
    }
    
    /**
     * Gets the mean of each feature.
     * 
     * @return the mean values
     */
    public double[] getMean() {
        if (!fitted) {
            throw new IllegalStateException("PCA must be fitted first");
        }
        return mean.clone();
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

