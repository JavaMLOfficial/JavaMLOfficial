package com.javaml.tree;

import com.javaml.array.NDArray;
import com.javaml.base.BaseEstimator;
import com.javaml.base.Estimator;

import java.util.*;

/**
 * Decision Tree Classifier.
 * Equivalent to scikit-learn's DecisionTreeClassifier.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class DecisionTreeClassifier extends BaseEstimator implements Estimator {
    
    private TreeNode root;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private boolean fitted = false;
    
    /**
     * Creates a new DecisionTreeClassifier with default parameters.
     */
    public DecisionTreeClassifier() {
        this.maxDepth = Integer.MAX_VALUE;
        this.minSamplesSplit = 2;
        this.minSamplesLeaf = 1;
    }
    
    /**
     * Creates a new DecisionTreeClassifier with specified parameters.
     * 
     * @param maxDepth maximum depth of the tree
     * @param minSamplesSplit minimum samples required to split a node
     * @param minSamplesLeaf minimum samples required in a leaf node
     */
    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit, int minSamplesLeaf) {
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
        return calculateAccuracy(predictions, y);
    }
    
    /**
     * Builds the decision tree recursively.
     */
    private TreeNode buildTree(NDArray X, NDArray y, List<Integer> indices, int depth) {
        // Check stopping criteria
        if (depth >= maxDepth || indices.size() < minSamplesSplit) {
            return new TreeNode(getMajorityClass(y, indices));
        }
        
        // Check if all samples have the same class
        if (isPure(y, indices)) {
            return new TreeNode(y.get(indices.get(0)));
        }
        
        // Find best split
        SplitResult bestSplit = findBestSplit(X, y, indices);
        
        if (bestSplit == null) {
            return new TreeNode(getMajorityClass(y, indices));
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
            return new TreeNode(getMajorityClass(y, indices));
        }
        
        // Recursively build children
        TreeNode node = new TreeNode(bestSplit.featureIndex, bestSplit.threshold);
        node.left = buildTree(X, y, leftIndices, depth + 1);
        node.right = buildTree(X, y, rightIndices, depth + 1);
        
        return node;
    }
    
    /**
     * Finds the best split for the given data.
     */
    private SplitResult findBestSplit(NDArray X, NDArray y, List<Integer> indices) {
        int nFeatures = X.getShape()[1];
        double bestGini = Double.MAX_VALUE;
        SplitResult bestSplit = null;
        
        for (int feature = 0; feature < nFeatures; feature++) {
            // Get unique values for this feature
            Set<Double> uniqueValues = new TreeSet<>();
            for (int idx : indices) {
                uniqueValues.add(X.get(idx, feature));
            }
            
            // Try each threshold
            List<Double> sortedValues = new ArrayList<>(uniqueValues);
            for (int i = 0; i < sortedValues.size() - 1; i++) {
                double threshold = (sortedValues.get(i) + sortedValues.get(i + 1)) / 2.0;
                
                // Calculate Gini impurity for this split
                double gini = calculateGiniForSplit(X, y, indices, feature, threshold);
                
                if (gini < bestGini) {
                    bestGini = gini;
                    bestSplit = new SplitResult(feature, threshold);
                }
            }
        }
        
        return bestSplit;
    }
    
    /**
     * Calculates Gini impurity for a split.
     */
    private double calculateGiniForSplit(NDArray X, NDArray y, List<Integer> indices, 
                                         int feature, double threshold) {
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        for (int idx : indices) {
            if (X.get(idx, feature) <= threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }
        
        if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
            return Double.MAX_VALUE;
        }
        
        double leftGini = calculateGini(y, leftIndices);
        double rightGini = calculateGini(y, rightIndices);
        
        double leftWeight = (double) leftIndices.size() / indices.size();
        double rightWeight = (double) rightIndices.size() / indices.size();
        
        return leftWeight * leftGini + rightWeight * rightGini;
    }
    
    /**
     * Calculates Gini impurity for a set of indices.
     */
    private double calculateGini(NDArray y, List<Integer> indices) {
        Map<Double, Integer> classCounts = new HashMap<>();
        for (int idx : indices) {
            double label = y.get(idx);
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        
        double gini = 1.0;
        int total = indices.size();
        
        for (int count : classCounts.values()) {
            double prob = (double) count / total;
            gini -= prob * prob;
        }
        
        return gini;
    }
    
    /**
     * Checks if all samples have the same class.
     */
    private boolean isPure(NDArray y, List<Integer> indices) {
        if (indices.isEmpty()) return true;
        
        double firstClass = y.get(indices.get(0));
        for (int idx : indices) {
            if (y.get(idx) != firstClass) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Gets the majority class for a set of indices.
     */
    private double getMajorityClass(NDArray y, List<Integer> indices) {
        Map<Double, Integer> classCounts = new HashMap<>();
        for (int idx : indices) {
            double label = y.get(idx);
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        
        double majorityClass = 0.0;
        int maxCount = 0;
        
        for (Map.Entry<Double, Integer> entry : classCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityClass = entry.getKey();
            }
        }
        
        return majorityClass;
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
        double value; // For leaf nodes
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

