package com.javaml.model_selection;

import com.javaml.array.NDArray;
import com.javaml.base.Estimator;

import java.util.Random;

/**
 * Model selection utilities including train_test_split and cross-validation.
 * Equivalent to scikit-learn's model_selection module.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ModelSelection {
    
    private ModelSelection() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Splits arrays into random train and test subsets.
     * 
     * @param X the features
     * @param y the targets
     * @param testSize the proportion of the dataset to include in the test split
     * @return an array containing [X_train, X_test, y_train, y_test]
     */
    public static NDArray[] trainTestSplit(NDArray X, NDArray y, double testSize) {
        return trainTestSplit(X, y, testSize, null);
    }
    
    /**
     * Splits arrays into random train and test subsets with a random state.
     * 
     * @param X the features
     * @param y the targets
     * @param testSize the proportion of the dataset to include in the test split
     * @param randomState the random seed
     * @return an array containing [X_train, X_test, y_train, y_test]
     */
    public static NDArray[] trainTestSplit(NDArray X, NDArray y, double testSize, Long randomState) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input arrays cannot be null");
        }
        if (X.getNdims() != 2) {
            throw new IllegalArgumentException("X must be 2D");
        }
        if (y.getNdims() != 1) {
            throw new IllegalArgumentException("y must be 1D");
        }
        if (X.getShape()[0] != y.getSize()) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (testSize <= 0.0 || testSize >= 1.0) {
            throw new IllegalArgumentException("testSize must be between 0 and 1");
        }
        
        int nSamples = X.getShape()[0];
        int nTest = (int) Math.round(nSamples * testSize);
        int nTrain = nSamples - nTest;
        
        // Create shuffled indices
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = i;
        }
        
        Random random = randomState != null ? new Random(randomState) : new Random();
        shuffleArray(indices, random);
        
        // Split indices
        int[] trainIndices = new int[nTrain];
        int[] testIndices = new int[nTest];
        System.arraycopy(indices, 0, trainIndices, 0, nTrain);
        System.arraycopy(indices, nTrain, testIndices, 0, nTest);
        
        // Create train/test splits
        NDArray X_train = selectRows(X, trainIndices);
        NDArray X_test = selectRows(X, testIndices);
        NDArray y_train = selectElements(y, trainIndices);
        NDArray y_test = selectElements(y, testIndices);
        
        return new NDArray[]{X_train, X_test, y_train, y_test};
    }
    
    /**
     * Performs k-fold cross-validation.
     * 
     * @param estimator the estimator to evaluate
     * @param X the features
     * @param y the targets
     * @param cv the number of folds
     * @return an array of scores for each fold
     */
    public static double[] crossValScore(Estimator estimator, NDArray X, NDArray y, int cv) {
        if (estimator == null) {
            throw new IllegalArgumentException("Estimator cannot be null");
        }
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input arrays cannot be null");
        }
        if (cv < 2) {
            throw new IllegalArgumentException("cv must be at least 2");
        }
        
        int nSamples = X.getShape()[0];
        int foldSize = nSamples / cv;
        
        double[] scores = new double[cv];
        
        for (int fold = 0; fold < cv; fold++) {
            int testStart = fold * foldSize;
            int testEnd = (fold == cv - 1) ? nSamples : (fold + 1) * foldSize;
            
            // Create train/test indices
            int[] trainIndices = new int[nSamples - (testEnd - testStart)];
            int[] testIndices = new int[testEnd - testStart];
            
            int trainIdx = 0;
            int testIdx = 0;
            for (int i = 0; i < nSamples; i++) {
                if (i >= testStart && i < testEnd) {
                    testIndices[testIdx++] = i;
                } else {
                    trainIndices[trainIdx++] = i;
                }
            }
            
            // Split data
            NDArray X_train = selectRows(X, trainIndices);
            NDArray X_test = selectRows(X, testIndices);
            NDArray y_train = selectElements(y, trainIndices);
            NDArray y_test = selectElements(y, testIndices);
            
            // Train and evaluate
            estimator.fit(X_train, y_train);
            scores[fold] = estimator.score(X_test, y_test);
        }
        
        return scores;
    }
    
    /**
     * Selects rows from a 2D array.
     */
    private static NDArray selectRows(NDArray X, int[] indices) {
        int[] shape = X.getShape();
        int nRows = indices.length;
        int nCols = shape[1];
        
        NDArray result = new NDArray(nRows, nCols);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                result.set(X.get(indices[i], j), i, j);
            }
        }
        return result;
    }
    
    /**
     * Selects elements from a 1D array.
     */
    private static NDArray selectElements(NDArray y, int[] indices) {
        double[] result = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = y.get(indices[i]);
        }
        return new NDArray(result, result.length);
    }
    
    /**
     * Shuffles an array using Fisher-Yates algorithm.
     */
    private static void shuffleArray(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

