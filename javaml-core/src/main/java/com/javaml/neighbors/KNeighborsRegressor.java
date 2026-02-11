package com.javaml.neighbors;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * K-Nearest Neighbors Regressor.
 * Equivalent to scikit-learn's KNeighborsRegressor.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class KNeighborsRegressor extends BaseEstimator implements Estimator {
    
    private int nNeighbors;
    private String weights; // "uniform" or "distance"
    private NDArray X_train;
    private NDArray y_train;
    private boolean fitted = false;
    
    /**
     * Creates a new KNeighborsRegressor with default parameters.
     */
    public KNeighborsRegressor() {
        this(5, "uniform");
    }
    
    /**
     * Creates a new KNeighborsRegressor with specified parameters.
     * 
     * @param nNeighbors number of neighbors
     * @param weights weight function ("uniform" or "distance")
     */
    public KNeighborsRegressor(int nNeighbors, String weights) {
        if (nNeighbors <= 0) {
            throw new IllegalArgumentException("nNeighbors must be positive");
        }
        this.nNeighbors = nNeighbors;
        this.weights = weights;
    }
    
    @Override
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
        if (shape[0] != y.getSize()) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        // Store training data (lazy learner)
        this.X_train = new NDArray(X);
        this.y_train = new NDArray(y);
        
        fitted = true;
    }
    
    @Override
    public NDArray predict(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        double[] predictions = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            predictions[i] = predictSingle(X, i);
        }
        
        return new NDArray(predictions, predictions.length);
    }
    
    @Override
    public double score(NDArray X, NDArray y) {
        NDArray predictions = predict(X);
        
        // Calculate R²
        double ssRes = 0.0;
        double ssTot = 0.0;
        double yMean = 0.0;
        
        for (int i = 0; i < y.getSize(); i++) {
            yMean += y.get(i);
        }
        yMean /= y.getSize();
        
        for (int i = 0; i < y.getSize(); i++) {
            double residual = y.get(i) - predictions.get(i);
            ssRes += residual * residual;
            double total = y.get(i) - yMean;
            ssTot += total * total;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Predicts a single sample.
     */
    private double predictSingle(NDArray X, int sampleIndex) {
        // Find k nearest neighbors
        List<Neighbor> neighbors = findNeighbors(X, sampleIndex);
        
        // Weighted average
        double sum = 0.0;
        double totalWeight = 0.0;
        
        for (Neighbor neighbor : neighbors) {
            double weight = "distance".equals(weights) ? 
                1.0 / (neighbor.distance + 1e-10) : 1.0;
            
            sum += neighbor.label * weight;
            totalWeight += weight;
        }
        
        return totalWeight > 0 ? sum / totalWeight : 0.0;
    }
    
    /**
     * Finds k nearest neighbors.
     */
    private List<Neighbor> findNeighbors(NDArray X, int sampleIndex) {
        int nTrain = X_train.getShape()[0];
        int nFeatures = X_train.getShape()[1];
        
        List<Neighbor> neighbors = new ArrayList<>();
        
        for (int i = 0; i < nTrain; i++) {
            double distance = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                double diff = X.get(sampleIndex, j) - X_train.get(i, j);
                distance += diff * diff;
            }
            distance = Math.sqrt(distance);
            
            neighbors.add(new Neighbor(i, distance, y_train.get(i)));
        }
        
        // Sort by distance and take k nearest
        neighbors.sort(Comparator.comparingDouble(n -> n.distance));
        return neighbors.subList(0, Math.min(nNeighbors, neighbors.size()));
    }
    
    /**
     * Helper class for storing neighbor information.
     */
    private static class Neighbor {
        int index;
        double distance;
        double label;
        
        Neighbor(int index, double distance, double label) {
            this.index = index;
            this.distance = distance;
            this.label = label;
        }
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("n_neighbors", nNeighbors);
        params.put("weights", weights);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_neighbors")) {
            this.nNeighbors = ((Number) params.get("n_neighbors")).intValue();
        }
        if (params.containsKey("weights")) {
            this.weights = (String) params.get("weights");
        }
    }
}

