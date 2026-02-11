package com.javaml.preprocessing;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Select K best features based on univariate statistical tests.
 * Equivalent to scikit-learn's SelectKBest.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class SelectKBest extends BaseEstimator implements Transformer {
    
    private int k;
    private String scoreFunc; // "f_regression" or "f_classif"
    private int[] featureIndices; // Indices of selected features
    private double[] scores; // Scores for all features
    private boolean fitted = false;
    
    /**
     * Creates a new SelectKBest with default k=10.
     */
    public SelectKBest() {
        this(10, "f_regression");
    }
    
    /**
     * Creates a new SelectKBest with specified parameters.
     * 
     * @param k number of features to select
     * @param scoreFunc scoring function ("f_regression" or "f_classif")
     */
    public SelectKBest(int k, String scoreFunc) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        this.k = k;
        this.scoreFunc = scoreFunc;
    }
    
    /**
     * Fits the selector to the data.
     * 
     * @param X the input features
     * @param y the target values
     */
    public void fit(NDArray X, NDArray y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D (samples x features)");
        }
        if (y.getNdims() != 1) {
            throw new IllegalArgumentException("y must be 1D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nSamples != y.getSize()) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        if (k > nFeatures) {
            throw new IllegalArgumentException("k cannot exceed number of features");
        }
        
        // Compute scores for all features
        scores = new double[nFeatures];
        
        for (int feature = 0; feature < nFeatures; feature++) {
            double[] featureData = new double[nSamples];
            for (int sample = 0; sample < nSamples; sample++) {
                featureData[sample] = X.get(sample, feature);
            }
            NDArray featureArray = new NDArray(featureData, nSamples);
            
            if ("f_regression".equals(scoreFunc)) {
                scores[feature] = fRegressionScore(featureArray, y);
            } else if ("f_classif".equals(scoreFunc)) {
                scores[feature] = fClassificationScore(featureArray, y);
            } else {
                throw new IllegalArgumentException("Unknown score function: " + scoreFunc);
            }
        }
        
        // Select top k features
        List<FeatureScore> featureScores = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++) {
            featureScores.add(new FeatureScore(i, scores[i]));
        }
        
        featureScores.sort((a, b) -> Double.compare(b.score, a.score));
        
        featureIndices = new int[k];
        for (int i = 0; i < k; i++) {
            featureIndices[i] = featureScores.get(i).index;
        }
        
        Arrays.sort(featureIndices);
        
        fitted = true;
    }
    
    /**
     * Computes F-score for regression.
     */
    private double fRegressionScore(NDArray X, NDArray y) {
        // Compute correlation coefficient
        double xMean = ArrayStats.mean(X);
        double yMean = ArrayStats.mean(y);
        
        double numerator = 0.0;
        double xVar = 0.0;
        double yVar = 0.0;
        
        for (int i = 0; i < X.getSize(); i++) {
            double xDiff = X.get(i) - xMean;
            double yDiff = y.get(i) - yMean;
            numerator += xDiff * yDiff;
            xVar += xDiff * xDiff;
            yVar += yDiff * yDiff;
        }
        
        if (xVar == 0.0 || yVar == 0.0) {
            return 0.0;
        }
        
        double correlation = numerator / Math.sqrt(xVar * yVar);
        double rSquared = correlation * correlation;
        
        // F-score = r^2 / (1 - r^2) * (n - 2)
        if (rSquared >= 1.0) {
            return Double.MAX_VALUE;
        }
        
        int n = X.getSize();
        return (rSquared / (1.0 - rSquared)) * (n - 2);
    }
    
    /**
     * Computes F-score for classification.
     */
    private double fClassificationScore(NDArray X, NDArray y) {
        // Get unique classes
        Set<Double> classes = new HashSet<>();
        for (int i = 0; i < y.getSize(); i++) {
            classes.add(y.get(i));
        }
        
        if (classes.size() < 2) {
            return 0.0;
        }
        
        // Compute between-class and within-class variance
        double xMean = ArrayStats.mean(X);
        double betweenVar = 0.0;
        double withinVar = 0.0;
        
        for (Double cls : classes) {
            List<Double> classValues = new ArrayList<>();
            for (int i = 0; i < y.getSize(); i++) {
                if (y.get(i) == cls) {
                    classValues.add(X.get(i));
                }
            }
            
            double classMean = 0.0;
            for (Double value : classValues) {
                classMean += value;
            }
            classMean /= classValues.size();
            
            double classVar = 0.0;
            for (Double value : classValues) {
                double diff = value - classMean;
                classVar += diff * diff;
            }
            
            betweenVar += classValues.size() * (classMean - xMean) * (classMean - xMean);
            withinVar += classVar;
        }
        
        if (withinVar == 0.0) {
            return betweenVar > 0.0 ? Double.MAX_VALUE : 0.0;
        }
        
        int nClasses = classes.size();
        int nSamples = X.getSize();
        
        double betweenMean = betweenVar / (nClasses - 1);
        double withinMean = withinVar / (nSamples - nClasses);
        
        return betweenMean / withinMean;
    }
    
    /**
     * Fits the selector (required by Transformer interface).
     * Note: SelectKBest requires both X and y, so use fit(X, y) instead.
     */
    @Override
    public void fit(NDArray X) {
        throw new UnsupportedOperationException(
            "SelectKBest requires both X and y. Use fit(X, y) instead.");
    }
    
    /**
     * Transforms the data.
     */
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("SelectKBest must be fitted before transform");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("Input must be 2D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        
        if (shape[1] != scores.length) {
            throw new IllegalArgumentException(
                "Number of features does not match fitted selector");
        }
        
        // Create result with only selected features
        NDArray result = new NDArray(nSamples, k);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < k; j++) {
                result.set(X.get(i, featureIndices[j]), i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Gets the scores for all features.
     * 
     * @return array of scores
     */
    public double[] getScores() {
        if (!fitted) {
            throw new IllegalStateException("SelectKBest must be fitted first");
        }
        return scores.clone();
    }
    
    /**
     * Gets the indices of selected features.
     * 
     * @return array of feature indices
     */
    public int[] getFeatureIndices() {
        if (!fitted) {
            throw new IllegalStateException("SelectKBest must be fitted first");
        }
        return featureIndices.clone();
    }
    
    /**
     * Helper class for feature scores.
     */
    private static class FeatureScore {
        int index;
        double score;
        
        FeatureScore(int index, double score) {
            this.index = index;
            this.score = score;
        }
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("k", k);
        params.put("score_func", scoreFunc);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("k")) {
            this.k = ((Number) params.get("k")).intValue();
        }
        if (params.containsKey("score_func")) {
            this.scoreFunc = (String) params.get("score_func");
        }
    }
}

