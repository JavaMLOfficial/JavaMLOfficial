package com.javaml.preprocessing;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Generates polynomial and interaction features.
 * Equivalent to scikit-learn's PolynomialFeatures.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class PolynomialFeatures extends BaseEstimator implements Transformer {
    
    private int degree;
    private boolean includeBias;
    private boolean interactionOnly;
    private int nInputFeatures;
    private int nOutputFeatures;
    private boolean fitted = false;
    
    /**
     * Creates a new PolynomialFeatures with default parameters.
     */
    public PolynomialFeatures() {
        this(2, true, false);
    }
    
    /**
     * Creates a new PolynomialFeatures with specified parameters.
     * 
     * @param degree the degree of the polynomial
     * @param includeBias if true, include a bias column
     * @param interactionOnly if true, only include interaction features
     */
    public PolynomialFeatures(int degree, boolean includeBias, boolean interactionOnly) {
        if (degree < 1) {
            throw new IllegalArgumentException("Degree must be at least 1");
        }
        this.degree = degree;
        this.includeBias = includeBias;
        this.interactionOnly = interactionOnly;
    }
    
    @Override
    public void fit(NDArray X) {
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D (samples x features)");
        }
        
        nInputFeatures = X.getShape()[1];
        nOutputFeatures = calculateOutputFeatures(nInputFeatures, degree, includeBias, interactionOnly);
        
        fitted = true;
    }
    
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("PolynomialFeatures must be fitted before transform");
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
        
        if (nFeatures != nInputFeatures) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted transformer");
        }
        
        List<double[]> featureList = new ArrayList<>();
        
        // Generate all polynomial combinations
        generatePolynomialFeatures(X, featureList, 0, new int[degree], 0);
        
        // Convert to array
        NDArray result = new NDArray(nSamples, nOutputFeatures);
        int colIdx = 0;
        
        for (double[] feature : featureList) {
            for (int i = 0; i < nSamples; i++) {
                result.set(feature[i], i, colIdx);
            }
            colIdx++;
        }
        
        return result;
    }
    
    /**
     * Generates polynomial features recursively.
     */
    private void generatePolynomialFeatures(NDArray X, List<double[]> featureList,
                                            int startIdx, int[] powers, int currentDegree) {
        if (currentDegree == 0) {
            // Bias term
            if (includeBias) {
                double[] bias = new double[X.getShape()[0]];
                Arrays.fill(bias, 1.0);
                featureList.add(bias);
            }
            generatePolynomialFeatures(X, featureList, 0, powers, 1);
            return;
        }
        
        if (currentDegree > degree) {
            return;
        }
        
        int nSamples = X.getShape()[0];
        int nFeatures = X.getShape()[1];
        
        for (int i = startIdx; i < nFeatures; i++) {
            powers[currentDegree - 1] = i;
            
            if (currentDegree == degree || (interactionOnly && currentDegree > 1)) {
                // Create feature
                double[] feature = new double[nSamples];
                for (int j = 0; j < nSamples; j++) {
                    double product = 1.0;
                    for (int k = 0; k < currentDegree; k++) {
                        product *= X.get(j, powers[k]);
                    }
                    feature[j] = product;
                }
                featureList.add(feature);
            }
            
            if (currentDegree < degree) {
                generatePolynomialFeatures(X, featureList, i, powers, currentDegree + 1);
            }
        }
    }
    
    /**
     * Calculates the number of output features.
     */
    private int calculateOutputFeatures(int nInput, int degree, boolean includeBias, 
                                       boolean interactionOnly) {
        int count = includeBias ? 1 : 0;
        
        if (interactionOnly) {
            // Only interactions: C(n, 2) + C(n, 3) + ... + C(n, degree)
            for (int d = 2; d <= degree; d++) {
                count += combination(nInput, d);
            }
        } else {
            // All combinations with repetition
            for (int d = 1; d <= degree; d++) {
                count += combinationWithRepetition(nInput, d);
            }
        }
        
        return count;
    }
    
    /**
     * Calculates combination C(n, k).
     */
    private int combination(int n, int k) {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;
        
        int result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
    
    /**
     * Calculates combination with repetition: C(n+k-1, k).
     */
    private int combinationWithRepetition(int n, int k) {
        return combination(n + k - 1, k);
    }
    
    /**
     * Gets the number of input features.
     * 
     * @return the number of input features
     */
    public int getNInputFeatures() {
        if (!fitted) {
            throw new IllegalStateException("Transformer must be fitted first");
        }
        return nInputFeatures;
    }
    
    /**
     * Gets the number of output features.
     * 
     * @return the number of output features
     */
    public int getNOutputFeatures() {
        if (!fitted) {
            throw new IllegalStateException("Transformer must be fitted first");
        }
        return nOutputFeatures;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("degree", degree);
        params.put("include_bias", includeBias);
        params.put("interaction_only", interactionOnly);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("degree")) {
            this.degree = ((Number) params.get("degree")).intValue();
        }
        if (params.containsKey("include_bias")) {
            this.includeBias = (Boolean) params.get("include_bias");
        }
        if (params.containsKey("interaction_only")) {
            this.interactionOnly = (Boolean) params.get("interaction_only");
        }
    }
}

