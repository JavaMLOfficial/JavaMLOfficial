package com.javaml.cluster;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.random.RandomGenerator;

import java.util.*;

/**
 * K-Means clustering algorithm.
 * Equivalent to scikit-learn's KMeans.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class KMeans extends BaseEstimator {
    
    private int nClusters;
    private int maxIterations;
    private double tolerance;
    private RandomGenerator random;
    private NDArray clusterCenters;
    private int[] labels;
    private double inertia;
    private boolean fitted = false;
    
    /**
     * Creates a new KMeans with default parameters.
     */
    public KMeans() {
        this(8, 300, 1e-4);
    }
    
    /**
     * Creates a new KMeans with specified parameters.
     * 
     * @param nClusters number of clusters
     * @param maxIterations maximum number of iterations
     * @param tolerance convergence tolerance
     */
    public KMeans(int nClusters, int maxIterations, double tolerance) {
        if (nClusters <= 0) {
            throw new IllegalArgumentException("nClusters must be positive");
        }
        this.nClusters = nClusters;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.random = new RandomGenerator();
    }
    
    /**
     * Fits the KMeans model to the data.
     * 
     * @param X the input data (2D: samples x features)
     */
    public void fit(NDArray X) {
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D (samples x features)");
        }
        
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        if (nSamples < nClusters) {
            throw new IllegalArgumentException(
                "Number of samples must be at least nClusters");
        }
        
        // Initialize cluster centers using k-means++ initialization
        clusterCenters = initializeCenters(X, nClusters);
        labels = new int[nSamples];
        
        // K-means algorithm
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Assign points to nearest centers
            int[] newLabels = assignLabels(X, clusterCenters);
            
            // Check for convergence
            boolean converged = true;
            for (int i = 0; i < nSamples; i++) {
                if (newLabels[i] != labels[i]) {
                    converged = false;
                    break;
                }
            }
            
            labels = newLabels;
            
            if (converged) {
                break;
            }
            
            // Update cluster centers
            NDArray newCenters = updateCenters(X, labels, nClusters);
            
            // Check if centers moved significantly
            double maxMovement = 0.0;
            for (int i = 0; i < nClusters; i++) {
                double movement = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    double diff = newCenters.get(i, j) - clusterCenters.get(i, j);
                    movement += diff * diff;
                }
                maxMovement = Math.max(maxMovement, Math.sqrt(movement));
            }
            
            clusterCenters = newCenters;
            
            if (maxMovement < tolerance) {
                break;
            }
        }
        
        // Calculate inertia (sum of squared distances to nearest center)
        inertia = calculateInertia(X, labels, clusterCenters);
        
        fitted = true;
    }
    
    /**
     * Predicts the closest cluster for each sample.
     * 
     * @param X the input data
     * @return array of cluster labels
     */
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
        
        int[] labels = assignLabels(X, clusterCenters);
        double[] labelArray = new double[labels.length];
        for (int i = 0; i < labels.length; i++) {
            labelArray[i] = labels[i];
        }
        return new NDArray(labelArray, labelArray.length);
    }
    
    /**
     * Gets the cluster centers.
     * 
     * @return the cluster centers
     */
    public NDArray getClusterCenters() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return new NDArray(clusterCenters);
    }
    
    /**
     * Gets the labels from the last fit.
     * 
     * @return the cluster labels
     */
    public int[] getLabels() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return labels.clone();
    }
    
    /**
     * Gets the inertia (sum of squared distances).
     * 
     * @return the inertia
     */
    public double getInertia() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return inertia;
    }
    
    /**
     * Initializes cluster centers using k-means++ algorithm.
     */
    private NDArray initializeCenters(NDArray X, int nClusters) {
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        NDArray centers = new NDArray(nClusters, nFeatures);
        
        // First center: random sample
        int firstIdx = random.randint(0, nSamples).get(0);
        for (int j = 0; j < nFeatures; j++) {
            centers.set(X.get(firstIdx, j), 0, j);
        }
        
        // Subsequent centers: choose with probability proportional to distance^2
        double[] distances = new double[nSamples];
        
        for (int i = 1; i < nClusters; i++) {
            // Calculate distances to nearest existing center
            for (int sample = 0; sample < nSamples; sample++) {
                double minDist = Double.MAX_VALUE;
                for (int center = 0; center < i; center++) {
                    double dist = 0.0;
                    for (int feature = 0; feature < nFeatures; feature++) {
                        double diff = X.get(sample, feature) - centers.get(center, feature);
                        dist += diff * diff;
                    }
                    minDist = Math.min(minDist, dist);
                }
                distances[sample] = minDist;
            }
            
            // Choose next center with probability proportional to distance^2
            double totalDistance = 0.0;
            for (double dist : distances) {
                totalDistance += dist;
            }
            
            double randomValue = random.rand(1).get(0) * totalDistance;
            double cumulative = 0.0;
            int chosenIdx = 0;
            
            for (int sample = 0; sample < nSamples; sample++) {
                cumulative += distances[sample];
                if (cumulative >= randomValue) {
                    chosenIdx = sample;
                    break;
                }
            }
            
            // Set new center
            for (int j = 0; j < nFeatures; j++) {
                centers.set(X.get(chosenIdx, j), i, j);
            }
        }
        
        return centers;
    }
    
    /**
     * Assigns each sample to the nearest cluster center.
     */
    private int[] assignLabels(NDArray X, NDArray centers) {
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        int nClusters = centers.getShape()[0];
        
        int[] labels = new int[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            double minDist = Double.MAX_VALUE;
            int nearestCluster = 0;
            
            for (int cluster = 0; cluster < nClusters; cluster++) {
                double dist = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    double diff = X.get(i, j) - centers.get(cluster, j);
                    dist += diff * diff;
                }
                
                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = cluster;
                }
            }
            
            labels[i] = nearestCluster;
        }
        
        return labels;
    }
    
    /**
     * Updates cluster centers based on assigned labels.
     */
    private NDArray updateCenters(NDArray X, int[] labels, int nClusters) {
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        NDArray newCenters = new NDArray(nClusters, nFeatures);
        int[] counts = new int[nClusters];
        
        // Sum points in each cluster
        for (int i = 0; i < nSamples; i++) {
            int cluster = labels[i];
            counts[cluster]++;
            for (int j = 0; j < nFeatures; j++) {
                double current = newCenters.get(cluster, j);
                newCenters.set(current + X.get(i, j), cluster, j);
            }
        }
        
        // Average to get new centers
        for (int cluster = 0; cluster < nClusters; cluster++) {
            if (counts[cluster] > 0) {
                for (int j = 0; j < nFeatures; j++) {
                    double sum = newCenters.get(cluster, j);
                    newCenters.set(sum / counts[cluster], cluster, j);
                }
            }
        }
        
        return newCenters;
    }
    
    /**
     * Calculates inertia (sum of squared distances).
     */
    private double calculateInertia(NDArray X, int[] labels, NDArray centers) {
        int[] shape = X.getShape();
        int nSamples = shape[0];
        int nFeatures = shape[1];
        
        double inertia = 0.0;
        
        for (int i = 0; i < nSamples; i++) {
            int cluster = labels[i];
            double dist = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                double diff = X.get(i, j) - centers.get(cluster, j);
                dist += diff * diff;
            }
            inertia += dist;
        }
        
        return inertia;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("n_clusters", nClusters);
        params.put("max_iterations", maxIterations);
        params.put("tolerance", tolerance);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_clusters")) {
            this.nClusters = ((Number) params.get("n_clusters")).intValue();
        }
        if (params.containsKey("max_iterations")) {
            this.maxIterations = ((Number) params.get("max_iterations")).intValue();
        }
        if (params.containsKey("tolerance")) {
            this.tolerance = ((Number) params.get("tolerance")).doubleValue();
        }
    }
}

