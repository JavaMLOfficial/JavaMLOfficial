package com.javaml.tree;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * Decision Tree Regressor.
 * Equivalent to scikit-learn's DecisionTreeRegressor.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class DecisionTreeRegressor extends BaseEstimator implements Estimator {
    
    private TreeNode root;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private boolean fitted = false;
    
    /**
     * Creates a new DecisionTreeRegressor with default parameters.
     */
    public DecisionTreeRegressor() {
        this.maxDepth = Integer.MAX_VALUE;
        this.minSamplesSplit = 2;
        this.minSamplesLeaf = 1;
    }
    
    /**
     * Creates a new DecisionTreeRegressor with specified parameters.
     * 
     * @param maxDepth maximum depth of the tree
     * @param minSamplesSplit minimum samples required to split a node
     * @param minSamplesLeaf minimum samples required in a leaf node
     */
    public DecisionTreeRegressor(int maxDepth, int minSamplesSplit, int minSamplesLeaf) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
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
        int nSamples = shape[0];
        
        if (nSamples != y.getSize()) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        // Build tree
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        
        root = buildTree(X, y, indices, 0);
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
        
        int nSamples = X.getShape()[0];
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
     * Builds the decision tree recursively.
     */
    private TreeNode buildTree(NDArray X, NDArray y, List<Integer> indices, int depth) {
        // Check stopping criteria
        if (depth >= maxDepth || indices.size() < minSamplesSplit) {
            return new TreeNode(getMean(y, indices));
        }
        
        // Check if variance is very low (all values similar)
        double variance = calculateVariance(y, indices);
        if (variance < 1e-10) {
            return new TreeNode(getMean(y, indices));
        }
        
        // Find best split
        SplitResult bestSplit = findBestSplit(X, y, indices);
        
        if (bestSplit == null) {
            return new TreeNode(getMean(y, indices));
        }
        
        // Split data
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        for (int idx : indices) {
            if (X.get(idx, bestSplit.featureIndex) <= bestSplit.threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }
        
        // Check minimum samples in leaf
        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            return new TreeNode(getMean(y, indices));
        }
        
        // Recursively build children
        TreeNode node = new TreeNode(bestSplit.featureIndex, bestSplit.threshold);
        node.left = buildTree(X, y, leftIndices, depth + 1);
        node.right = buildTree(X, y, rightIndices, depth + 1);
        
        return node;
    }
    
    /**
     * Finds the best split for the given data using MSE reduction.
     */
    private SplitResult findBestSplit(NDArray X, NDArray y, List<Integer> indices) {
        int nFeatures = X.getShape()[1];
        double bestMSE = Double.MAX_VALUE;
        SplitResult bestSplit = null;
        
        for (int feature = 0; feature < nFeatures; feature++) {
            // Get unique values for this feature
            Set<Double> uniqueValues = new TreeSet<>();
            for (int idx : indices) {
                uniqueValues.add(X.get(idx, feature));
            }
            
            // Try each threshold
            for (double threshold : uniqueValues) {
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                
                for (int idx : indices) {
                    if (X.get(idx, feature) <= threshold) {
                        left.add(idx);
                    } else {
                        right.add(idx);
                    }
                }
                
                if (left.isEmpty() || right.isEmpty()) {
                    continue;
                }
                
                // Calculate MSE for this split
                double mse = calculateMSE(y, left, right);
                
                if (mse < bestMSE) {
                    bestMSE = mse;
                    bestSplit = new SplitResult(feature, threshold);
                }
            }
        }
        
        return bestSplit;
    }
    
    /**
     * Calculates mean squared error for a split.
     */
    private double calculateMSE(NDArray y, List<Integer> left, List<Integer> right) {
        double leftMean = getMean(y, left);
        double rightMean = getMean(y, right);
        
        double leftMSE = 0.0;
        for (int idx : left) {
            double diff = y.get(idx) - leftMean;
            leftMSE += diff * diff;
        }
        leftMSE /= left.size();
        
        double rightMSE = 0.0;
        for (int idx : right) {
            double diff = y.get(idx) - rightMean;
            rightMSE += diff * diff;
        }
        rightMSE /= right.size();
        
        // Weighted average
        double totalSize = left.size() + right.size();
        return (left.size() * leftMSE + right.size() * rightMSE) / totalSize;
    }
    
    /**
     * Calculates variance of values.
     */
    private double calculateVariance(NDArray y, List<Integer> indices) {
        if (indices.isEmpty()) return 0.0;
        
        double mean = getMean(y, indices);
        double variance = 0.0;
        
        for (int idx : indices) {
            double diff = y.get(idx) - mean;
            variance += diff * diff;
        }
        
        return variance / indices.size();
    }
    
    /**
     * Gets the mean value for a set of indices.
     */
    private double getMean(NDArray y, List<Integer> indices) {
        if (indices.isEmpty()) return 0.0;
        
        double sum = 0.0;
        for (int idx : indices) {
            sum += y.get(idx);
        }
        return sum / indices.size();
    }
    
    /**
     * Predicts a single sample.
     */
    private double predictSingle(NDArray X, int sampleIndex) {
        TreeNode node = root;
        
        while (!node.isLeaf()) {
            if (X.get(sampleIndex, node.featureIndex) <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        
        return node.value;
    }
    
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("max_depth", maxDepth);
        params.put("min_samples_split", minSamplesSplit);
        params.put("min_samples_leaf", minSamplesLeaf);
        params.put("fitted", fitted);
        return params;
    }
    
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("max_depth")) {
            this.maxDepth = ((Number) params.get("max_depth")).intValue();
        }
        if (params.containsKey("min_samples_split")) {
            this.minSamplesSplit = ((Number) params.get("min_samples_split")).intValue();
        }
        if (params.containsKey("min_samples_leaf")) {
            this.minSamplesLeaf = ((Number) params.get("min_samples_leaf")).intValue();
        }
    }
    
    /**
     * Internal tree node class.
     */
    private static class TreeNode {
        int featureIndex;
        double threshold;
        double value; // For leaf nodes (mean value)
        TreeNode left;
        TreeNode right;
        
        TreeNode(double value) {
            this.value = value;
        }
        
        TreeNode(int featureIndex, double threshold) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
        }
        
        boolean isLeaf() {
            return left == null && right == null;
        }
    }
    
    /**
     * Result of a split operation.
     */
    private static class SplitResult {
        int featureIndex;
        double threshold;
        
        SplitResult(int featureIndex, double threshold) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
        }
    }
}

