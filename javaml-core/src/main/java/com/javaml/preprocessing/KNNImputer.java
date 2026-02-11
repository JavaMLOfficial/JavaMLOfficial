package com.javaml.preprocessing;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;
import com.javaml.neighbors.KNeighborsRegressor;

import java.util.*;

/**
 * Imputation for completing missing values using k-nearest neighbors.
 * Equivalent to scikit-learn's KNNImputer.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class KNNImputer extends BaseEstimator implements Transformer {
    
    private int nNeighbors;
    private KNeighborsRegressor[] regressors;
    private boolean fitted = false;
    
    /**
     * Creates a new KNNImputer with default parameters.
     */
    public KNNImputer() {
        this(5);
    }
    
    /**
     * Creates a new KNNImputer with specified number of neighbors.
     * 
     * @param nNeighbors number of neighbors to use
     */
    public KNNImputer(int nNeighbors) {
        if (nNeighbors <= 0) {
            throw new IllegalArgumentException("nNeighbors must be positive");
        }
        this.nNeighbors = nNeighbors;
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
        
        regressors = new KNeighborsRegressor[nFeatures];
        
        // Train a KNN regressor for each feature
        for (int feature = 0; feature < nFeatures; feature++) {
            // Find samples with non-missing values for this feature
            List<Integer> validIndices = new ArrayList<>();
            for (int sample = 0; sample < nSamples; sample++) {
                if (!Double.isNaN(X.get(sample, feature))) {
                    validIndices.add(sample);
                }
            }
            
            if (validIndices.isEmpty()) {
                // All values are missing - skip this feature
                regressors[feature] = null;
                continue;
            }
            
            // Create training data (other features as X, this feature as y)
            int nValid = validIndices.size();
            NDArray XTrain = new NDArray(nValid, nFeatures - 1);
            NDArray yTrain = new NDArray(nValid);
            
            int trainIdx = 0;
            for (int idx : validIndices) {
                int colIdx = 0;
                for (int f = 0; f < nFeatures; f++) {
                    if (f != feature) {
                        XTrain.set(X.get(idx, f), trainIdx, colIdx++);
                    }
                }
                yTrain.set(X.get(idx, feature), trainIdx);
                trainIdx++;
            }
            
            // Train KNN regressor
            KNeighborsRegressor regressor = new KNeighborsRegressor(nNeighbors, "distance");
            regressor.fit(XTrain, yTrain);
            regressors[feature] = regressor;
        }
        
        fitted = true;
    }
    
    @Override
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Imputer must be fitted before transform");
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
        
        if (nFeatures != regressors.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted imputer");
        }
        
        NDArray result = new NDArray(X);
        
        // Impute missing values for each feature
        for (int feature = 0; feature < nFeatures; feature++) {
            if (regressors[feature] == null) {
                continue; // Skip if all values were missing during fit
            }
            
            // Find samples with missing values for this feature
            List<Integer> missingIndices = new ArrayList<>();
            for (int sample = 0; sample < nSamples; sample++) {
                if (Double.isNaN(result.get(sample, feature))) {
                    missingIndices.add(sample);
                }
            }
            
            if (missingIndices.isEmpty()) {
                continue;
            }
            
            // Create prediction data
            NDArray XPred = new NDArray(missingIndices.size(), nFeatures - 1);
            int predIdx = 0;
            for (int idx : missingIndices) {
                int colIdx = 0;
                for (int f = 0; f < nFeatures; f++) {
                    if (f != feature) {
                        XPred.set(result.get(idx, f), predIdx, colIdx++);
                    }
                }
                predIdx++;
            }
            
            // Predict missing values
            NDArray predictions = regressors[feature].predict(XPred);
            
            // Fill in missing values
            predIdx = 0;
            for (int idx : missingIndices) {
                result.set(predictions.get(predIdx++), idx, feature);
            }
        }
        
        return result;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("n_neighbors", nNeighbors);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_neighbors")) {
            this.nNeighbors = ((Number) params.get("n_neighbors")).intValue();
        }
    }
}

