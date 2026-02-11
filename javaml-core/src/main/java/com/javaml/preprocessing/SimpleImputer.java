package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.HashMap;
import java.util.Map;

/**
 * Simple imputation transformer for completing missing values.
 * Equivalent to scikit-learn's SimpleImputer.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class SimpleImputer extends BaseEstimator implements Transformer {
    
    public enum Strategy {
        MEAN, MEDIAN, MOST_FREQUENT, CONSTANT
    }
    
    private Strategy strategy;
    private double fillValue;
    private double[] statistics;
    private boolean fitted = false;
    
    /**
     * Creates a new SimpleImputer with mean strategy.
     */
    public SimpleImputer() {
        this(Strategy.MEAN, 0.0);
    }
    
    /**
     * Creates a new SimpleImputer with specified strategy.
     * 
     * @param strategy the imputation strategy
     */
    public SimpleImputer(Strategy strategy) {
        this(strategy, 0.0);
    }
    
    /**
     * Creates a new SimpleImputer with constant strategy.
     * 
     * @param fillValue the constant value to use for imputation
     */
    public SimpleImputer(double fillValue) {
        this(Strategy.CONSTANT, fillValue);
    }
    
    /**
     * Creates a new SimpleImputer with specified strategy and fill value.
     * 
     * @param strategy the imputation strategy
     * @param fillValue the constant value (used only for CONSTANT strategy)
     */
    public SimpleImputer(Strategy strategy, double fillValue) {
        this.strategy = strategy;
        this.fillValue = fillValue;
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
        
        statistics = new double[nFeatures];
        
        for (int feature = 0; feature < nFeatures; feature++) {
            // Collect non-NaN values for this feature
            double[] featureData = new double[nSamples];
            int validCount = 0;
            
            for (int sample = 0; sample < nSamples; sample++) {
                double value = X.get(sample, feature);
                if (!Double.isNaN(value)) {
                    featureData[validCount++] = value;
                }
            }
            
            if (validCount == 0) {
                // All values are NaN, use fillValue
                statistics[feature] = fillValue;
                continue;
            }
            
            // Calculate statistic based on strategy
            NDArray featureArray = new NDArray(featureData, validCount);
            
            switch (strategy) {
                case MEAN:
                    statistics[feature] = ArrayStats.mean(featureArray);
                    break;
                case MEDIAN:
                    statistics[feature] = ArrayStats.median(featureArray);
                    break;
                case MOST_FREQUENT:
                    statistics[feature] = calculateMostFrequent(featureArray);
                    break;
                case CONSTANT:
                    statistics[feature] = fillValue;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown strategy: " + strategy);
            }
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
        
        if (nFeatures != statistics.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted imputer");
        }
        
        NDArray result = new NDArray(nSamples, nFeatures);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = X.get(i, j);
                if (Double.isNaN(value)) {
                    result.set(statistics[j], i, j);
                } else {
                    result.set(value, i, j);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Calculates the most frequent value (mode).
     */
    private double calculateMostFrequent(NDArray array) {
        double[] data = array.getData();
        Map<Double, Integer> frequency = new HashMap<>();
        
        for (double value : data) {
            frequency.put(value, frequency.getOrDefault(value, 0) + 1);
        }
        
        double mostFrequent = data[0];
        int maxCount = 0;
        
        for (Map.Entry<Double, Integer> entry : frequency.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostFrequent = entry.getKey();
            }
        }
        
        return mostFrequent;
    }
    
    /**
     * Gets the imputation statistics.
     * 
     * @return array of statistics for each feature
     */
    public double[] getStatistics() {
        if (!fitted) {
            throw new IllegalStateException("Imputer must be fitted first");
        }
        return statistics.clone();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("strategy", strategy);
        params.put("fill_value", fillValue);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("strategy")) {
            this.strategy = (Strategy) params.get("strategy");
        }
        if (params.containsKey("fill_value")) {
            this.fillValue = ((Number) params.get("fill_value")).doubleValue();
        }
    }
}

