package com.javaml.random;

import com.javaml.array.ArrayCreation;
import com.javaml.array.NDArray;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Random number generation equivalent to NumPy's random module.
 * Provides functions for generating random numbers from various distributions.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class RandomGenerator {
    
    private final Random random;
    
    /**
     * Creates a new RandomGenerator with a default random number generator.
     */
    public RandomGenerator() {
        this.random = ThreadLocalRandom.current();
    }
    
    /**
     * Creates a new RandomGenerator with a seeded random number generator.
     * 
     * @param seed the seed value
     */
    public RandomGenerator(long seed) {
        this.random = new Random(seed);
    }
    
    /**
     * Creates a new RandomGenerator with the provided Random instance.
     * 
     * @param random the Random instance to use
     */
    public RandomGenerator(Random random) {
        if (random == null) {
            throw new IllegalArgumentException("Random cannot be null");
        }
        this.random = random;
    }
    
    /**
     * Generates random values in [0, 1) with the specified shape.
     * 
     * @param shape the shape of the array
     * @return an array of random values
     */
    public NDArray rand(int... shape) {
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            array.setFlat(i, random.nextDouble());
        }
        return array;
    }
    
    /**
     * Generates random values from a standard normal distribution.
     * 
     * @param shape the shape of the array
     * @return an array of random values
     */
    public NDArray randn(int... shape) {
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            array.setFlat(i, random.nextGaussian());
        }
        return array;
    }
    
    /**
     * Generates random integers in [low, high).
     * 
     * @param low the lower bound (inclusive)
     * @param high the upper bound (exclusive)
     * @param shape the shape of the array
     * @return an array of random integers
     */
    public NDArray randint(int low, int high, int... shape) {
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        int range = high - low;
        for (int i = 0; i < size; i++) {
            array.setFlat(i, low + random.nextInt(range));
        }
        return array;
    }
    
    /**
     * Generates random integers in [0, high).
     * 
     * @param high the upper bound (exclusive)
     * @param shape the shape of the array
     * @return an array of random integers
     */
    public NDArray randint(int high, int... shape) {
        return randint(0, high, shape);
    }
    
    /**
     * Generates random values from a uniform distribution [low, high).
     * 
     * @param low the lower bound
     * @param high the upper bound
     * @param shape the shape of the array
     * @return an array of random values
     */
    public NDArray uniform(double low, double high, int... shape) {
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        double range = high - low;
        for (int i = 0; i < size; i++) {
            array.setFlat(i, low + random.nextDouble() * range);
        }
        return array;
    }
    
    /**
     * Generates random values from a normal distribution.
     * 
     * @param mean the mean
     * @param std the standard deviation
     * @param shape the shape of the array
     * @return an array of random values
     */
    public NDArray normal(double mean, double std, int... shape) {
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            array.setFlat(i, mean + random.nextGaussian() * std);
        }
        return array;
    }
    
    /**
     * Generates random values from an exponential distribution.
     * 
     * @param scale the scale parameter (1/lambda)
     * @param shape the shape of the array
     * @return an array of random values
     */
    public NDArray exponential(double scale, int... shape) {
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            array.setFlat(i, -scale * Math.log(1 - random.nextDouble()));
        }
        return array;
    }
    
    /**
     * Generates random values from a beta distribution.
     * 
     * @param alpha the alpha parameter
     * @param beta the beta parameter
     * @param shape the shape of the array
     * @return an array of random values
     */
    public NDArray beta(double alpha, double beta, int... shape) {
        // Simplified implementation using acceptance-rejection
        // Full implementation would use proper beta distribution algorithm
        NDArray array = new NDArray(shape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            // Use gamma-based generation (simplified)
            double x = generateGamma(alpha, 1.0);
            double y = generateGamma(beta, 1.0);
            array.setFlat(i, x / (x + y));
        }
        return array;
    }
    
    /**
     * Helper method to generate a single gamma-distributed value.
     */
    private double generateGamma(double shape, double scale) {
        // Simplified implementation using acceptance-rejection
        double result = 0.0;
        for (int j = 0; j < (int) Math.ceil(shape); j++) {
            result -= Math.log(1 - random.nextDouble());
        }
        return result * scale;
    }
    
    /**
     * Generates random values from a gamma distribution.
     * 
     * @param shape the shape parameter
     * @param scale the scale parameter
     * @param outputShape the shape of the output array
     * @return an array of random values
     */
    public NDArray gamma(double shape, double scale, int... outputShape) {
        // Simplified implementation
        NDArray array = new NDArray(outputShape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            // Use acceptance-rejection method (simplified)
            double result = 0.0;
            for (int j = 0; j < (int) Math.ceil(shape); j++) {
                result -= Math.log(1 - random.nextDouble());
            }
            array.setFlat(i, result * scale);
        }
        return array;
    }
    
    /**
     * Generates random values from a Poisson distribution.
     * 
     * @param lambda the lambda parameter
     * @param outputShape the shape of the output array
     * @return an array of random values
     */
    public NDArray poisson(double lambda, int... outputShape) {
        NDArray array = new NDArray(outputShape);
        int size = array.getSize();
        for (int i = 0; i < size; i++) {
            int k = 0;
            double p = 1.0;
            double L = Math.exp(-lambda);
            do {
                k++;
                p *= random.nextDouble();
            } while (p > L);
            array.setFlat(i, k - 1);
        }
        return array;
    }
    
    /**
     * Randomly selects elements from an array.
     * 
     * @param array the source array
     * @param size the number of elements to select
     * @param replace whether to sample with replacement
     * @return an array of selected elements
     */
    public NDArray choice(NDArray array, int size, boolean replace) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        if (!replace && size > array.getSize()) {
            throw new IllegalArgumentException(
                "Cannot sample more elements than array size without replacement");
        }
        
        double[] source = array.getData();
        double[] result = new double[size];
        
        if (replace) {
            for (int i = 0; i < size; i++) {
                result[i] = source[random.nextInt(source.length)];
            }
        } else {
            // Fisher-Yates shuffle for sampling without replacement
            int[] indices = new int[source.length];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            
            for (int i = 0; i < size; i++) {
                int j = i + random.nextInt(indices.length - i);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
                result[i] = source[indices[i]];
            }
        }
        
        return new NDArray(result, result.length);
    }
    
    /**
     * Randomly shuffles an array in-place.
     * 
     * @param array the array to shuffle
     */
    public void shuffle(NDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        double[] data = array.getData();
        for (int i = data.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    /**
     * Generates a random permutation.
     * 
     * @param n the size of the permutation
     * @return an array with values 0 to n-1 in random order
     */
    public NDArray permutation(int n) {
        NDArray array = ArrayCreation.arange(0, n);
        shuffle(array);
        return array;
    }
    
    /**
     * Sets the seed for the random number generator.
     * 
     * @param seed the seed value
     */
    public void seed(long seed) {
        if (random instanceof Random) {
            ((Random) random).setSeed(seed);
        }
        // Note: ThreadLocalRandom doesn't support setSeed
        // This is a limitation - consider using a different Random implementation
    }
    
    /**
     * Gets a default shared instance of RandomGenerator.
     * 
     * @return a shared RandomGenerator instance
     */
    public static RandomGenerator getDefault() {
        return DefaultInstanceHolder.INSTANCE;
    }
    
    /**
     * Holder for the default shared instance.
     */
    private static class DefaultInstanceHolder {
        private static final RandomGenerator INSTANCE = new RandomGenerator();
    }
}

