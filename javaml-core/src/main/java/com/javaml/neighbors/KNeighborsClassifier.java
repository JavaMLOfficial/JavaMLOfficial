package com.javaml.neighbors;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * K-Nearest Neighbors Classifier.
 * Equivalent to scikit-learn's KNeighborsClassifier.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class KNeighborsClassifier extends BaseEstimator implements Estimator {
    
    private int nNeighbors;
    private String weights; // "uniform" or "distance"
    private NDArray X_train;
    private NDArray y_train;
    private boolean fitted = false;
    
    /**
     * Creates a new KNeighborsClassifier with default parameters.
     */
    public KNeighborsClassifier() {
        this(5, "uniform");
    }
    
    /**
     * Creates a new KNeighborsClassifier with specified parameters.
     * 
     * @param nNeighbors number of neighbors
     * @param weights weight function ("uniform" or "distance")
     */
    public KNeighborsClassifier(int nNeighbors, String weights) {
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
        return calculateAccuracy(predictions, y);
    }
    
    /**
     * Predicts a single sample.
     */
    private double predictSingle(NDArray X, int sampleIndex) {
        // Find k nearest neighbors
        List<Neighbor> neighbors = findNeighbors(X, sampleIndex);
        
        // Count votes
        Map<Double, Double> votes = new HashMap<>();
        
        for (Neighbor neighbor : neighbors) {
            double weight = "distance".equals(weights) ? 
                1.0 / (neighbor.distance + 1e-10) : 1.0;
            
            double label = neighbor.label;
            votes.put(label, votes.getOrDefault(label, 0.0) + weight);
        }
        
        // Find majority vote
        double majorityClass = 0.0;
        double maxVotes = 0.0;
        
        for (Map.Entry<Double, Double> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                majorityClass = entry.getKey();
            }
        }
        
        return majorityClass;
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
     * Calculates accuracy.
     */
    private double calculateAccuracy(NDArray predictions, NDArray y) {
        int correct = 0;
        for (int i = 0; i < predictions.getSize(); i++) {
            if (Math.abs(predictions.get(i) - y.get(i)) < 1e-6) {
                correct++;
            }
        }
        return (double) correct / predictions.getSize();
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

