package com.javaml.cluster;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;

import java.util.*;

/**
 * DBSCAN clustering algorithm.
 * Equivalent to scikit-learn's DBSCAN.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class DBSCAN extends BaseEstimator {
    
    private double eps;
    private int minSamples;
    private int[] labels;
    private boolean fitted = false;
    
    /**
     * Creates a new DBSCAN with default parameters.
     */
    public DBSCAN() {
        this(0.5, 5);
    }
    
    /**
     * Creates a new DBSCAN with specified parameters.
     * 
     * @param eps maximum distance between samples in the same neighborhood
     * @param minSamples minimum number of samples in a neighborhood
     */
    public DBSCAN(double eps, int minSamples) {
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive");
        }
        if (minSamples <= 0) {
            throw new IllegalArgumentException("minSamples must be positive");
        }
        this.eps = eps;
        this.minSamples = minSamples;
    }
    
    /**
     * Fits the DBSCAN model to the data.
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
        
        labels = new int[nSamples];
        Arrays.fill(labels, -1); // -1 means noise/unclassified
        
        int clusterId = 0;
        boolean[] visited = new boolean[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            if (visited[i]) {
                continue;
            }
            
            visited[i] = true;
            
            // Find neighbors
            List<Integer> neighbors = findNeighbors(X, i, eps);
            
            if (neighbors.size() < minSamples) {
                // Mark as noise
                labels[i] = -1;
            } else {
                // Start a new cluster
                labels[i] = clusterId;
                expandCluster(X, i, neighbors, clusterId, visited);
                clusterId++;
            }
        }
        
        fitted = true;
    }
    
    /**
     * Expands a cluster from a seed point.
     */
    private void expandCluster(NDArray X, int seedIdx, List<Integer> neighbors, 
                              int clusterId, boolean[] visited) {
        Queue<Integer> seeds = new LinkedList<>(neighbors);
        
        while (!seeds.isEmpty()) {
            int currentIdx = seeds.poll();
            
            if (labels[currentIdx] == -1) {
                labels[currentIdx] = clusterId;
            }
            
            if (!visited[currentIdx]) {
                visited[currentIdx] = true;
                List<Integer> currentNeighbors = findNeighbors(X, currentIdx, eps);
                
                if (currentNeighbors.size() >= minSamples) {
                    seeds.addAll(currentNeighbors);
                }
            }
        }
    }
    
    /**
     * Finds neighbors within eps distance.
     */
    private List<Integer> findNeighbors(NDArray X, int pointIdx, double eps) {
        List<Integer> neighbors = new ArrayList<>();
        int nSamples = X.getShape()[0];
        int nFeatures = X.getShape()[1];
        
        for (int i = 0; i < nSamples; i++) {
            if (i == pointIdx) {
                continue;
            }
            
            double distance = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                double diff = X.get(pointIdx, j) - X.get(i, j);
                distance += diff * diff;
            }
            distance = Math.sqrt(distance);
            
            if (distance <= eps) {
                neighbors.add(i);
            }
        }
        
        return neighbors;
    }
    
    /**
     * Predicts cluster labels (same as fit for DBSCAN).
     * 
     * @param X the input data
     * @return cluster labels
     */
    public NDArray predict(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        
        // DBSCAN doesn't predict new points - it only clusters training data
        // For new points, we could assign to nearest cluster, but that's not standard DBSCAN
        // For now, return the labels from fit
        double[] labelArray = new double[labels.length];
        for (int i = 0; i < labels.length; i++) {
            labelArray[i] = labels[i];
        }
        return new NDArray(labelArray, labelArray.length);
    }
    
    /**
     * Gets the cluster labels.
     * 
     * @return the cluster labels (-1 for noise)
     */
    public int[] getLabels() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return labels.clone();
    }
    
    /**
     * Gets the number of clusters found (excluding noise).
     * 
     * @return the number of clusters
     */
    public int getNClusters() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        
        Set<Integer> clusters = new HashSet<>();
        for (int label : labels) {
            if (label != -1) {
                clusters.add(label);
            }
        }
        return clusters.size();
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("eps", eps);
        params.put("min_samples", minSamples);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("eps")) {
            this.eps = ((Number) params.get("eps")).doubleValue();
        }
        if (params.containsKey("min_samples")) {
            this.minSamples = ((Number) params.get("min_samples")).intValue();
        }
    }
}

