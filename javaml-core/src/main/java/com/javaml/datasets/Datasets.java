package com.javaml.datasets;

import com.javaml.array.ArrayCreation;
import com.javaml.array.ArrayManipulation;
import com.javaml.array.NDArray;
import com.javaml.random.RandomGenerator;

/**
 * Dataset generation and loading utilities.
 * Equivalent to scikit-learn's datasets module.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class Datasets {
    
    private static final RandomGenerator random = RandomGenerator.getDefault();
    
    private Datasets() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Generate a random n-class classification problem.
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param nClasses number of classes
     * @param nInformative number of informative features
     * @param nRedundant number of redundant features
     * @param nClustersPerClass number of clusters per class
     * @param randomState random seed
     * @return array containing [X, y]
     */
    public static NDArray[] makeClassification(int nSamples, int nFeatures, int nClasses,
                                                int nInformative, int nRedundant, 
                                                int nClustersPerClass, Long randomState) {
        if (nSamples <= 0 || nFeatures <= 0 || nClasses <= 0) {
            throw new IllegalArgumentException("All parameters must be positive");
        }
        if (nInformative + nRedundant > nFeatures) {
            throw new IllegalArgumentException(
                "nInformative + nRedundant cannot exceed nFeatures");
        }
        
        RandomGenerator rng = randomState != null ? new RandomGenerator(randomState) : random;
        
        // Generate informative features
        NDArray X = rng.randn(nSamples, nInformative);
        
        // Generate redundant features (linear combinations of informative)
        if (nRedundant > 0) {
            NDArray redundant = new NDArray(nSamples, nRedundant);
            for (int i = 0; i < nRedundant; i++) {
                int informativeIdx = rng.randint(0, nInformative).get(0);
                for (int j = 0; j < nSamples; j++) {
                    redundant.set(X.get(j, informativeIdx) + rng.randn(1).get(0) * 0.1, j, i);
                }
            }
            X = ArrayManipulation.concatenate(new NDArray[]{X, redundant}, 1);
        }
        
        // Generate remaining features (noise)
        int nRemaining = nFeatures - nInformative - nRedundant;
        if (nRemaining > 0) {
            NDArray remaining = rng.randn(nSamples, nRemaining);
            X = ArrayManipulation.concatenate(new NDArray[]{X, remaining}, 1);
        }
        
        // Generate class labels
        double[] y = new double[nSamples];
        int samplesPerClass = nSamples / nClasses;
        
        for (int classIdx = 0; classIdx < nClasses; classIdx++) {
            int start = classIdx * samplesPerClass;
            int end = (classIdx == nClasses - 1) ? nSamples : (classIdx + 1) * samplesPerClass;
            
            for (int i = start; i < end; i++) {
                y[i] = classIdx;
            }
        }
        
        // Shuffle
        for (int i = nSamples - 1; i > 0; i--) {
            int j = rng.randint(0, i + 1).get(0);
            double temp = y[i];
            y[i] = y[j];
            y[j] = temp;
            
            // Also shuffle corresponding row in X
            for (int k = 0; k < nFeatures; k++) {
                double tempX = X.get(i, k);
                X.set(X.get(j, k), i, k);
                X.set(tempX, j, k);
            }
        }
        
        return new NDArray[]{X, new NDArray(y, y.length)};
    }
    
    /**
     * Generate a random n-class classification problem (default parameters).
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param nClasses number of classes
     * @return array containing [X, y]
     */
    public static NDArray[] makeClassification(int nSamples, int nFeatures, int nClasses) {
        int nInformative = Math.min(nFeatures, nClasses * 2);
        int nRedundant = Math.max(0, nFeatures - nInformative) / 2;
        return makeClassification(nSamples, nFeatures, nClasses, nInformative, nRedundant, 1, null);
    }
    
    /**
     * Generate a random regression problem.
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param nInformative number of informative features
     * @param noise standard deviation of Gaussian noise
     * @param randomState random seed
     * @return array containing [X, y]
     */
    public static NDArray[] makeRegression(int nSamples, int nFeatures, int nInformative,
                                          double noise, Long randomState) {
        if (nSamples <= 0 || nFeatures <= 0) {
            throw new IllegalArgumentException("nSamples and nFeatures must be positive");
        }
        if (nInformative > nFeatures) {
            throw new IllegalArgumentException("nInformative cannot exceed nFeatures");
        }
        
        RandomGenerator rng = randomState != null ? new RandomGenerator(randomState) : random;
        
        // Generate features
        NDArray X = rng.randn(nSamples, nFeatures);
        
        // Generate target (linear combination of informative features)
        double[] y = new double[nSamples];
        double[] coefficients = new double[nInformative];
        for (int i = 0; i < nInformative; i++) {
            coefficients[i] = rng.randn(1).get(0);
        }
        
        for (int i = 0; i < nSamples; i++) {
            double sum = 0.0;
            for (int j = 0; j < nInformative; j++) {
                sum += coefficients[j] * X.get(i, j);
            }
            y[i] = sum + rng.randn(1).get(0) * noise;
        }
        
        return new NDArray[]{X, new NDArray(y, y.length)};
    }
    
    /**
     * Generate a random regression problem (default parameters).
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @return array containing [X, y]
     */
    public static NDArray[] makeRegression(int nSamples, int nFeatures) {
        return makeRegression(nSamples, nFeatures, nFeatures, 0.0, null);
    }
    
    /**
     * Generate isotropic Gaussian blobs for clustering.
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param centers number of centers (or array of centers)
     * @param clusterStd standard deviation of clusters
     * @param randomState random seed
     * @return array containing [X, y] where y are cluster labels
     */
    public static NDArray[] makeBlobs(int nSamples, int nFeatures, int centers,
                                     double clusterStd, Long randomState) {
        if (nSamples <= 0 || nFeatures <= 0 || centers <= 0) {
            throw new IllegalArgumentException("All parameters must be positive");
        }
        
        RandomGenerator rng = randomState != null ? new RandomGenerator(randomState) : random;
        
        // Generate centers
        NDArray centerArray = rng.randn(centers, nFeatures).multiply(10.0);
        
        // Generate samples
        int samplesPerCenter = nSamples / centers;
        NDArray X = new NDArray(nSamples, nFeatures);
        double[] y = new double[nSamples];
        
        int sampleIdx = 0;
        for (int centerIdx = 0; centerIdx < centers; centerIdx++) {
            int nSamplesForCenter = (centerIdx == centers - 1) ? 
                (nSamples - sampleIdx) : samplesPerCenter;
            
            for (int i = 0; i < nSamplesForCenter; i++) {
                y[sampleIdx] = centerIdx;
                for (int j = 0; j < nFeatures; j++) {
                    double value = centerArray.get(centerIdx, j) + 
                                  rng.randn(1).get(0) * clusterStd;
                    X.set(value, sampleIdx, j);
                }
                sampleIdx++;
            }
        }
        
        // Shuffle
        for (int i = nSamples - 1; i > 0; i--) {
            int j = rng.randint(0, i + 1).get(0);
            double temp = y[i];
            y[i] = y[j];
            y[j] = temp;
            
            for (int k = 0; k < nFeatures; k++) {
                double tempX = X.get(i, k);
                X.set(X.get(j, k), i, k);
                X.set(tempX, j, k);
            }
        }
        
        return new NDArray[]{X, new NDArray(y, y.length)};
    }
    
    /**
     * Generate isotropic Gaussian blobs (default parameters).
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param centers number of centers
     * @return array containing [X, y]
     */
    public static NDArray[] makeBlobs(int nSamples, int nFeatures, int centers) {
        return makeBlobs(nSamples, nFeatures, centers, 1.0, null);
    }
    
    /**
     * Generate two interleaving half circles (moons).
     * 
     * @param nSamples number of samples
     * @param noise standard deviation of Gaussian noise
     * @param randomState random seed
     * @return array containing [X, y]
     */
    public static NDArray[] makeMoons(int nSamples, double noise, Long randomState) {
        if (nSamples <= 0) {
            throw new IllegalArgumentException("nSamples must be positive");
        }
        
        RandomGenerator rng = randomState != null ? new RandomGenerator(randomState) : random;
        
        NDArray X = new NDArray(nSamples, 2);
        double[] y = new double[nSamples];
        
        int samplesPerMoon = nSamples / 2;
        
        for (int i = 0; i < samplesPerMoon; i++) {
            double angle = Math.PI * i / samplesPerMoon;
            double x = Math.cos(angle);
            double y_coord = Math.sin(angle);
            
            X.set(x + rng.randn(1).get(0) * noise, i, 0);
            X.set(y_coord + rng.randn(1).get(0) * noise, i, 1);
            y[i] = 0.0;
        }
        
        for (int i = 0; i < samplesPerMoon; i++) {
            double angle = Math.PI * i / samplesPerMoon;
            double x = 1.0 - Math.cos(angle);
            double y_coord = 1.0 - Math.sin(angle) - 0.5;
            
            X.set(x + rng.randn(1).get(0) * noise, samplesPerMoon + i, 0);
            X.set(y_coord + rng.randn(1).get(0) * noise, samplesPerMoon + i, 1);
            y[samplesPerMoon + i] = 1.0;
        }
        
        return new NDArray[]{X, new NDArray(y, y.length)};
    }
    
    /**
     * Generate two interleaving half circles (default parameters).
     * 
     * @param nSamples number of samples
     * @return array containing [X, y]
     */
    public static NDArray[] makeMoons(int nSamples) {
        return makeMoons(nSamples, 0.1, null);
    }
    
    /**
     * Generate two concentric circles.
     * 
     * @param nSamples number of samples
     * @param noise standard deviation of Gaussian noise
     * @param factor scale factor between inner and outer circle
     * @param randomState random seed
     * @return array containing [X, y]
     */
    public static NDArray[] makeCircles(int nSamples, double noise, double factor,
                                       Long randomState) {
        if (nSamples <= 0) {
            throw new IllegalArgumentException("nSamples must be positive");
        }
        if (factor <= 0 || factor >= 1) {
            throw new IllegalArgumentException("factor must be between 0 and 1");
        }
        
        RandomGenerator rng = randomState != null ? new RandomGenerator(randomState) : random;
        
        NDArray X = new NDArray(nSamples, 2);
        double[] y = new double[nSamples];
        
        int samplesPerCircle = nSamples / 2;
        
        // Inner circle
        for (int i = 0; i < samplesPerCircle; i++) {
            double angle = 2 * Math.PI * i / samplesPerCircle;
            double radius = factor;
            double x = radius * Math.cos(angle) + rng.randn(1).get(0) * noise;
            double y_coord = radius * Math.sin(angle) + rng.randn(1).get(0) * noise;
            
            X.set(x, i, 0);
            X.set(y_coord, i, 1);
            y[i] = 0.0;
        }
        
        // Outer circle
        for (int i = 0; i < samplesPerCircle; i++) {
            double angle = 2 * Math.PI * i / samplesPerCircle;
            double radius = 1.0;
            double x = radius * Math.cos(angle) + rng.randn(1).get(0) * noise;
            double y_coord = radius * Math.sin(angle) + rng.randn(1).get(0) * noise;
            
            X.set(x, samplesPerCircle + i, 0);
            X.set(y_coord, samplesPerCircle + i, 1);
            y[samplesPerCircle + i] = 1.0;
        }
        
        return new NDArray[]{X, new NDArray(y, y.length)};
    }
    
    /**
     * Generate two concentric circles (default parameters).
     * 
     * @param nSamples number of samples
     * @return array containing [X, y]
     */
    public static NDArray[] makeCircles(int nSamples) {
        return makeCircles(nSamples, 0.1, 0.5, null);
    }
}

