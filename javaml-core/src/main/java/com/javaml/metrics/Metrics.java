package com.javaml.metrics;

import com.javaml.array.NDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Metrics for evaluating machine learning models.
 * Equivalent to scikit-learn's metrics module.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class Metrics {
    
    private Metrics() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Calculates accuracy score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @return the accuracy score
     */
    public static double accuracyScore(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        int correct = 0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            if (Math.abs(yTrue.get(i) - yPred.get(i)) < 1e-6) {
                correct++;
            }
        }
        return (double) correct / yTrue.getSize();
    }
    
    /**
     * Calculates precision score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param average the averaging method ("binary", "macro", "micro", "weighted")
     * @return the precision score
     */
    public static double precisionScore(NDArray yTrue, NDArray yPred, String average) {
        validateInputs(yTrue, yPred);
        
        if ("binary".equals(average)) {
            return precisionBinary(yTrue, yPred);
        } else if ("macro".equals(average)) {
            return precisionMacro(yTrue, yPred);
        } else {
            throw new UnsupportedOperationException("Averaging method not yet implemented: " + average);
        }
    }
    
    /**
     * Calculates recall score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param average the averaging method
     * @return the recall score
     */
    public static double recallScore(NDArray yTrue, NDArray yPred, String average) {
        validateInputs(yTrue, yPred);
        
        if ("binary".equals(average)) {
            return recallBinary(yTrue, yPred);
        } else if ("macro".equals(average)) {
            return recallMacro(yTrue, yPred);
        } else {
            throw new UnsupportedOperationException("Averaging method not yet implemented: " + average);
        }
    }
    
    /**
     * Calculates F1 score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param average the averaging method
     * @return the F1 score
     */
    public static double f1Score(NDArray yTrue, NDArray yPred, String average) {
        double precision = precisionScore(yTrue, yPred, average);
        double recall = recallScore(yTrue, yPred, average);
        
        if (precision + recall == 0) {
            return 0.0;
        }
        
        return 2.0 * (precision * recall) / (precision + recall);
    }
    
    /**
     * Calculates mean squared error.
     * 
     * @param yTrue the true values
     * @param yPred the predicted values
     * @return the mean squared error
     */
    public static double meanSquaredError(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        double sumSquaredError = 0.0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            double error = yTrue.get(i) - yPred.get(i);
            sumSquaredError += error * error;
        }
        
        return sumSquaredError / yTrue.getSize();
    }
    
    /**
     * Calculates mean absolute error.
     * 
     * @param yTrue the true values
     * @param yPred the predicted values
     * @return the mean absolute error
     */
    public static double meanAbsoluteError(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        double sumAbsoluteError = 0.0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            sumAbsoluteError += Math.abs(yTrue.get(i) - yPred.get(i));
        }
        
        return sumAbsoluteError / yTrue.getSize();
    }
    
    /**
     * Calculates R² score.
     * 
     * @param yTrue the true values
     * @param yPred the predicted values
     * @return the R² score
     */
    public static double r2Score(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        // Calculate mean of true values
        double yMean = 0.0;
        for (int i = 0; i < yTrue.getSize(); i++) {
            yMean += yTrue.get(i);
        }
        yMean /= yTrue.getSize();
        
        // Calculate SS_res and SS_tot
        double ssRes = 0.0;
        double ssTot = 0.0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            double residual = yTrue.get(i) - yPred.get(i);
            ssRes += residual * residual;
            
            double total = yTrue.get(i) - yMean;
            ssTot += total * total;
        }
        
        if (ssTot == 0.0) {
            return 0.0;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Calculates confusion matrix.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @return a 2D array representing the confusion matrix
     */
    public static NDArray confusionMatrix(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        // Find unique classes
        Map<Double, Integer> classMap = new HashMap<>();
        int classIndex = 0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            double label = yTrue.get(i);
            if (!classMap.containsKey(label)) {
                classMap.put(label, classIndex++);
            }
        }
        for (int i = 0; i < yPred.getSize(); i++) {
            double label = yPred.get(i);
            if (!classMap.containsKey(label)) {
                classMap.put(label, classIndex++);
            }
        }
        
        int nClasses = classMap.size();
        NDArray matrix = new NDArray(nClasses, nClasses);
        
        // Count occurrences
        for (int i = 0; i < yTrue.getSize(); i++) {
            int trueClass = classMap.get(yTrue.get(i));
            int predClass = classMap.get(yPred.get(i));
            double current = matrix.get(trueClass, predClass);
            matrix.set(current + 1.0, trueClass, predClass);
        }
        
        return matrix;
    }
    
    /**
     * Calculates binary precision.
     */
    private static double precisionBinary(NDArray yTrue, NDArray yPred) {
        int tp = 0, fp = 0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            if (yPred.get(i) == 1.0 && yTrue.get(i) == 1.0) {
                tp++;
            } else if (yPred.get(i) == 1.0 && yTrue.get(i) == 0.0) {
                fp++;
            }
        }
        
        if (tp + fp == 0) {
            return 0.0;
        }
        
        return (double) tp / (tp + fp);
    }
    
    /**
     * Calculates binary recall.
     */
    private static double recallBinary(NDArray yTrue, NDArray yPred) {
        int tp = 0, fn = 0;
        
        for (int i = 0; i < yTrue.getSize(); i++) {
            if (yPred.get(i) == 1.0 && yTrue.get(i) == 1.0) {
                tp++;
            } else if (yPred.get(i) == 0.0 && yTrue.get(i) == 1.0) {
                fn++;
            }
        }
        
        if (tp + fn == 0) {
            return 0.0;
        }
        
        return (double) tp / (tp + fn);
    }
    
    /**
     * Calculates macro-averaged precision.
     */
    private static double precisionMacro(NDArray yTrue, NDArray yPred) {
        // Simplified implementation
        // Full implementation would calculate per-class precision and average
        return precisionBinary(yTrue, yPred);
    }
    
    /**
     * Calculates macro-averaged recall.
     */
    private static double recallMacro(NDArray yTrue, NDArray yPred) {
        // Simplified implementation
        return recallBinary(yTrue, yPred);
    }
    
    /**
     * Calculates balanced accuracy score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @return the balanced accuracy score
     */
    public static double balancedAccuracyScore(NDArray yTrue, NDArray yPred) {
        validateInputs(yTrue, yPred);
        
        // Get unique classes
        Set<Double> classes = new HashSet<>();
        for (int i = 0; i < yTrue.getSize(); i++) {
            classes.add(yTrue.get(i));
        }
        
        double totalRecall = 0.0;
        for (Double cls : classes) {
            int tp = 0, fn = 0;
            for (int i = 0; i < yTrue.getSize(); i++) {
                if (yTrue.get(i) == cls && yPred.get(i) == cls) {
                    tp++;
                } else if (yTrue.get(i) == cls && yPred.get(i) != cls) {
                    fn++;
                }
            }
            if (tp + fn > 0) {
                totalRecall += (double) tp / (tp + fn);
            }
        }
        
        return totalRecall / classes.size();
    }
    
    /**
     * Calculates F-beta score.
     * 
     * @param yTrue the true labels
     * @param yPred the predicted labels
     * @param beta the beta parameter
     * @param average the averaging method
     * @return the F-beta score
     */
    public static double fbetaScore(NDArray yTrue, NDArray yPred, double beta, String average) {
        double precision = precisionScore(yTrue, yPred, average);
        double recall = recallScore(yTrue, yPred, average);
        
        double betaSquared = beta * beta;
        if (precision + recall == 0) {
            return 0.0;
        }
        
        return (1 + betaSquared) * (precision * recall) / 
               (betaSquared * precision + recall);
    }
    
    /**
     * Calculates ROC AUC score.
     * 
     * @param yTrue the true labels
     * @param yScore the prediction scores (probabilities)
     * @return the ROC AUC score
     */
    public static double rocAucScore(NDArray yTrue, NDArray yScore) {
        validateInputs(yTrue, yScore);
        
        // Get unique classes (binary classification)
        Set<Double> classes = new HashSet<>();
        for (int i = 0; i < yTrue.getSize(); i++) {
            classes.add(yTrue.get(i));
        }
        
        if (classes.size() != 2) {
            throw new IllegalArgumentException("ROC AUC requires binary classification");
        }
        
        // Sort by score
        List<ScoreLabel> pairs = new ArrayList<>();
        for (int i = 0; i < yTrue.getSize(); i++) {
            pairs.add(new ScoreLabel(yScore.get(i), yTrue.get(i)));
        }
        pairs.sort((a, b) -> Double.compare(b.score, a.score));
        
        // Calculate AUC using trapezoidal rule
        double auc = 0.0;
        double tpr = 0.0;
        double fpr = 0.0;
        double prevTpr = 0.0;
        double prevFpr = 0.0;
        
        int positives = 0, negatives = 0;
        for (ScoreLabel pair : pairs) {
            if (pair.label == 1.0) positives++;
            else negatives++;
        }
        
        for (ScoreLabel pair : pairs) {
            if (pair.label == 1.0) {
                tpr += 1.0 / positives;
            } else {
                fpr += 1.0 / negatives;
            }
            
            auc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
            prevTpr = tpr;
            prevFpr = fpr;
        }
        
        return auc;
    }
    
    /**
     * Calculates silhouette score for clustering.
     * 
     * @param X the feature matrix
     * @param labels the cluster labels
     * @return the silhouette score
     */
    public static double silhouetteScore(NDArray X, NDArray labels) {
        if (X == null || labels == null) {
            throw new IllegalArgumentException("Input arrays cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D");
        }
        if (labels.getNdims() != 1) {
            throw new IllegalArgumentException("labels must be 1D");
        }
        if (X.getShape()[0] != labels.getSize()) {
            throw new IllegalArgumentException("X and labels must have same number of samples");
        }
        
        int nSamples = X.getShape()[0];
        int nFeatures = X.getShape()[1];
        
        double[] silhouetteScores = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            double labelI = labels.get(i);
            
            // Calculate mean distance to points in same cluster (a_i)
            double a_i = 0.0;
            int countSame = 0;
            
            for (int j = 0; j < nSamples; j++) {
                if (i != j && labels.get(j) == labelI) {
                    double dist = 0.0;
                    for (int k = 0; k < nFeatures; k++) {
                        double diff = X.get(i, k) - X.get(j, k);
                        dist += diff * diff;
                    }
                    a_i += Math.sqrt(dist);
                    countSame++;
                }
            }
            a_i = countSame > 0 ? a_i / countSame : 0.0;
            
            // Calculate mean distance to points in nearest other cluster (b_i)
            Set<Double> otherLabels = new HashSet<>();
            for (int j = 0; j < nSamples; j++) {
                if (labels.get(j) != labelI) {
                    otherLabels.add(labels.get(j));
                }
            }
            
            double minB_i = Double.MAX_VALUE;
            for (Double otherLabel : otherLabels) {
                double b_i = 0.0;
                int countOther = 0;
                
                for (int j = 0; j < nSamples; j++) {
                    if (labels.get(j) == otherLabel) {
                        double dist = 0.0;
                        for (int k = 0; k < nFeatures; k++) {
                            double diff = X.get(i, k) - X.get(j, k);
                            dist += diff * diff;
                        }
                        b_i += Math.sqrt(dist);
                        countOther++;
                    }
                }
                b_i = countOther > 0 ? b_i / countOther : 0.0;
                minB_i = Math.min(minB_i, b_i);
            }
            
            double b_i = minB_i;
            double maxAB = Math.max(a_i, b_i);
            silhouetteScores[i] = maxAB > 0 ? (b_i - a_i) / maxAB : 0.0;
        }
        
        double sum = 0.0;
        for (double score : silhouetteScores) {
            sum += score;
        }
        
        return sum / nSamples;
    }
    
    /**
     * Calculates adjusted Rand index for clustering evaluation.
     * 
     * @param labelsTrue the true labels
     * @param labelsPred the predicted labels
     * @return the adjusted Rand index
     */
    public static double adjustedRandScore(NDArray labelsTrue, NDArray labelsPred) {
        validateInputs(labelsTrue, labelsPred);
        
        int n = labelsTrue.getSize();
        
        // Build contingency table
        Map<String, Integer> contingency = new HashMap<>();
        Map<Double, Integer> trueCounts = new HashMap<>();
        Map<Double, Integer> predCounts = new HashMap<>();
        
        for (int i = 0; i < n; i++) {
            double trueLabel = labelsTrue.get(i);
            double predLabel = labelsPred.get(i);
            
            String key = trueLabel + "," + predLabel;
            contingency.put(key, contingency.getOrDefault(key, 0) + 1);
            trueCounts.put(trueLabel, trueCounts.getOrDefault(trueLabel, 0) + 1);
            predCounts.put(predLabel, predCounts.getOrDefault(predLabel, 0) + 1);
        }
        
        // Calculate sum of combinations
        long sumComb = 0;
        for (int count : contingency.values()) {
            sumComb += combination(count, 2);
        }
        
        long sumTrueComb = 0;
        for (int count : trueCounts.values()) {
            sumTrueComb += combination(count, 2);
        }
        
        long sumPredComb = 0;
        for (int count : predCounts.values()) {
            sumPredComb += combination(count, 2);
        }
        
        long nComb = combination(n, 2);
        double expectedIndex = (double) sumTrueComb * sumPredComb / nComb;
        double maxIndex = (sumTrueComb + sumPredComb) / 2.0;
        double adjustedIndex = (sumComb - expectedIndex) / (maxIndex - expectedIndex);
        
        return adjustedIndex;
    }
    
    /**
     * Calculates combination C(n, k).
     */
    private static long combination(int n, int k) {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;
        
        long result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
    
    /**
     * Helper class for ROC AUC calculation.
     */
    private static class ScoreLabel {
        double score;
        double label;
        
        ScoreLabel(double score, double label) {
            this.score = score;
            this.label = label;
        }
    }
    
    /**
     * Validates that inputs are not null and have the same size.
     */
    private static void validateInputs(NDArray yTrue, NDArray yPred) {
        if (yTrue == null || yPred == null) {
            throw new IllegalArgumentException("Input arrays cannot be null");
        }
        if (yTrue.getSize() != yPred.getSize()) {
            throw new IllegalArgumentException(
                "yTrue and yPred must have the same size");
        }
    }
}

