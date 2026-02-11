package com.javaml.array;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.DoubleStream;

/**
 * Statistical functions for NDArray, equivalent to NumPy's statistical routines.
 * Provides functions for computing mean, median, standard deviation, variance, and more.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ArrayStats {
    
    private ArrayStats() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Computes the mean of array elements.
     * 
     * @param array the input array
     * @return the mean value
     */
    public static double mean(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            return Double.NaN;
        }
        
        double[] data = array.getData();
        double sum = 0.0;
        for (double value : data) {
            sum += value;
        }
        return sum / data.length;
    }
    
    /**
     * Computes the median of array elements.
     * 
     * @param array the input array
     * @return the median value
     */
    public static double median(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            return Double.NaN;
        }
        
        double[] data = array.getData();
        double[] sorted = Arrays.copyOf(data, data.length);
        Arrays.sort(sorted);
        
        int mid = sorted.length / 2;
        if (sorted.length % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        } else {
            return sorted[mid];
        }
    }
    
    /**
     * Computes the standard deviation of array elements.
     * 
     * @param array the input array
     * @return the standard deviation
     */
    public static double std(NDArray array) {
        return std(array, true);
    }
    
    /**
     * Computes the standard deviation of array elements.
     * 
     * @param array the input array
     * @param ddof delta degrees of freedom (0 for population, 1 for sample)
     * @return the standard deviation
     */
    public static double std(NDArray array, boolean ddof) {
        return Math.sqrt(var(array, ddof));
    }
    
    /**
     * Computes the variance of array elements.
     * 
     * @param array the input array
     * @return the variance
     */
    public static double var(NDArray array) {
        return var(array, true);
    }
    
    /**
     * Computes the variance of array elements.
     * 
     * @param array the input array
     * @param ddof delta degrees of freedom (0 for population, 1 for sample)
     * @return the variance
     */
    public static double var(NDArray array, boolean ddof) {
        validateArray(array);
        if (array.getSize() == 0) {
            return Double.NaN;
        }
        
        double mean = mean(array);
        double[] data = array.getData();
        double sumSquaredDiff = 0.0;
        
        for (double value : data) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        
        int divisor = array.getSize() - (ddof ? 1 : 0);
        return divisor > 0 ? sumSquaredDiff / divisor : Double.NaN;
    }
    
    /**
     * Finds the minimum value in the array.
     * 
     * @param array the input array
     * @return the minimum value
     */
    public static double min(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            return Double.NaN;
        }
        
        double[] data = array.getData();
        double min = data[0];
        for (int i = 1; i < data.length; i++) {
            if (data[i] < min) {
                min = data[i];
            }
        }
        return min;
    }
    
    /**
     * Finds the maximum value in the array.
     * 
     * @param array the input array
     * @return the maximum value
     */
    public static double max(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            return Double.NaN;
        }
        
        double[] data = array.getData();
        double max = data[0];
        for (int i = 1; i < data.length; i++) {
            if (data[i] > max) {
                max = data[i];
            }
        }
        return max;
    }
    
    /**
     * Finds the index of the minimum value.
     * 
     * @param array the input array
     * @return the index of the minimum value
     */
    public static int argmin(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            throw new IllegalArgumentException("Cannot find argmin of empty array");
        }
        
        double[] data = array.getData();
        int minIndex = 0;
        double minValue = data[0];
        
        for (int i = 1; i < data.length; i++) {
            if (data[i] < minValue) {
                minValue = data[i];
                minIndex = i;
            }
        }
        return minIndex;
    }
    
    /**
     * Finds the index of the maximum value.
     * 
     * @param array the input array
     * @return the index of the maximum value
     */
    public static int argmax(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            throw new IllegalArgumentException("Cannot find argmax of empty array");
        }
        
        double[] data = array.getData();
        int maxIndex = 0;
        double maxValue = data[0];
        
        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxValue) {
                maxValue = data[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    /**
     * Computes the sum of array elements.
     * 
     * @param array the input array
     * @return the sum
     */
    public static double sum(NDArray array) {
        validateArray(array);
        double[] data = array.getData();
        double sum = 0.0;
        for (double value : data) {
            sum += value;
        }
        return sum;
    }
    
    /**
     * Computes the product of array elements.
     * 
     * @param array the input array
     * @return the product
     */
    public static double prod(NDArray array) {
        validateArray(array);
        if (array.getSize() == 0) {
            return 1.0;
        }
        
        double[] data = array.getData();
        double product = 1.0;
        for (double value : data) {
            product *= value;
        }
        return product;
    }
    
    /**
     * Computes the cumulative sum.
     * 
     * @param array the input array
     * @return a new array with cumulative sums
     */
    public static NDArray cumsum(NDArray array) {
        validateArray(array);
        double[] data = array.getData();
        double[] result = new double[data.length];
        
        if (data.length > 0) {
            result[0] = data[0];
            for (int i = 1; i < data.length; i++) {
                result[i] = result[i - 1] + data[i];
            }
        }
        
        return new NDArray(result, result.length);
    }
    
    /**
     * Computes the cumulative product.
     * 
     * @param array the input array
     * @return a new array with cumulative products
     */
    public static NDArray cumprod(NDArray array) {
        validateArray(array);
        double[] data = array.getData();
        double[] result = new double[data.length];
        
        if (data.length > 0) {
            result[0] = data[0];
            for (int i = 1; i < data.length; i++) {
                result[i] = result[i - 1] * data[i];
            }
        }
        
        return new NDArray(result, result.length);
    }
    
    /**
     * Computes the percentile of array elements.
     * 
     * @param array the input array
     * @param percentile the percentile (0-100)
     * @return the percentile value
     */
    public static double percentile(NDArray array, double percentile) {
        validateArray(array);
        if (array.getSize() == 0) {
            return Double.NaN;
        }
        
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("Percentile must be between 0 and 100");
        }
        
        double[] data = array.getData();
        double[] sorted = Arrays.copyOf(data, data.length);
        Arrays.sort(sorted);
        
        double index = (percentile / 100.0) * (sorted.length - 1);
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);
        
        if (lower == upper) {
            return sorted[lower];
        }
        
        double weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
    
    /**
     * Computes the quantile of array elements.
     * 
     * @param array the input array
     * @param q the quantile (0-1)
     * @return the quantile value
     */
    public static double quantile(NDArray array, double q) {
        return percentile(array, q * 100);
    }
    
    /**
     * Validates that an array is not null.
     */
    private static void validateArray(NDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
    }
}

