package com.javaml.array;

import java.util.Arrays;
import java.util.function.DoubleFunction;

/**
 * Array creation functions equivalent to NumPy's array creation routines.
 * Provides static factory methods for creating NDArrays with various initializations.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ArrayCreation {
    
    private ArrayCreation() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Creates an array from a sequence of values.
     * 
     * @param values the values to create the array from
     * @return a new NDArray containing the values
     */
    public static NDArray array(double... values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }
        return new NDArray(values, values.length);
    }
    
    /**
     * Creates an array from a sequence of values with the specified shape.
     * 
     * @param values the values to create the array from
     * @param shape the desired shape
     * @return a new NDArray with the specified shape
     */
    public static NDArray array(double[] values, int... shape) {
        if (values == null) {
            throw new IllegalArgumentException("Values cannot be null");
        }
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be null or empty");
        }
        return new NDArray(values, shape);
    }
    
    /**
     * Creates an array filled with zeros.
     * 
     * @param shape the dimensions of the array
     * @return a new NDArray filled with zeros
     */
    public static NDArray zeros(int... shape) {
        return new NDArray(shape);
    }
    
    /**
     * Creates an array filled with ones.
     * 
     * @param shape the dimensions of the array
     * @return a new NDArray filled with ones
     */
    public static NDArray ones(int... shape) {
        return zeros(shape).fill(1.0);
    }
    
    /**
     * Creates an empty array (uninitialized values).
     * 
     * @param shape the dimensions of the array
     * @return a new uninitialized NDArray
     */
    public static NDArray empty(int... shape) {
        return new NDArray(shape);
    }
    
    /**
     * Creates an array with values from start to stop (exclusive) with step size.
     * 
     * @param start the start value (inclusive)
     * @param stop the stop value (exclusive)
     * @param step the step size
     * @return a new 1D NDArray
     */
    public static NDArray arange(double start, double stop, double step) {
        if (step == 0) {
            throw new IllegalArgumentException("Step cannot be zero");
        }
        
        int size = (int) Math.ceil((stop - start) / step);
        if (size <= 0) {
            size = 0;
        }
        
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = start + i * step;
        }
        
        return new NDArray(data, size);
    }
    
    /**
     * Creates an array with values from 0 to stop (exclusive) with step 1.
     * 
     * @param stop the stop value (exclusive)
     * @return a new 1D NDArray
     */
    public static NDArray arange(double stop) {
        return arange(0, stop, 1);
    }
    
    /**
     * Creates an array with values from start to stop (exclusive) with step 1.
     * 
     * @param start the start value (inclusive)
     * @param stop the stop value (exclusive)
     * @return a new 1D NDArray
     */
    public static NDArray arange(double start, double stop) {
        return arange(start, stop, 1);
    }
    
    /**
     * Creates an array with evenly spaced values from start to stop (inclusive).
     * 
     * @param start the start value
     * @param stop the stop value (inclusive)
     * @param num the number of samples
     * @return a new 1D NDArray
     */
    public static NDArray linspace(double start, double stop, int num) {
        return linspace(start, stop, num, true);
    }
    
    /**
     * Creates an array with evenly spaced values from start to stop.
     * 
     * @param start the start value
     * @param stop the stop value
     * @param num the number of samples
     * @param endpoint whether to include the stop value
     * @return a new 1D NDArray
     */
    public static NDArray linspace(double start, double stop, int num, boolean endpoint) {
        if (num < 0) {
            throw new IllegalArgumentException("Number of samples must be non-negative");
        }
        if (num == 0) {
            return new NDArray(0);
        }
        if (num == 1) {
            return new NDArray(new double[]{start}, 1);
        }
        
        double[] data = new double[num];
        double step;
        if (endpoint) {
            step = (stop - start) / (num - 1);
        } else {
            step = (stop - start) / num;
        }
        
        for (int i = 0; i < num; i++) {
            data[i] = start + i * step;
        }
        
        return new NDArray(data, num);
    }
    
    /**
     * Creates an array with logarithmically spaced values.
     * 
     * @param start the start value (base^start)
     * @param stop the stop value (base^stop)
     * @param num the number of samples
     * @return a new 1D NDArray
     */
    public static NDArray logspace(double start, double stop, int num) {
        return logspace(start, stop, num, 10.0, true);
    }
    
    /**
     * Creates an array with logarithmically spaced values.
     * 
     * @param start the start value (base^start)
     * @param stop the stop value (base^stop)
     * @param num the number of samples
     * @param base the logarithmic base
     * @param endpoint whether to include the stop value
     * @return a new 1D NDArray
     */
    public static NDArray logspace(double start, double stop, int num, double base, boolean endpoint) {
        NDArray linear = linspace(start, stop, num, endpoint);
        double[] data = linear.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.pow(base, data[i]);
        }
        return new NDArray(data, data.length);
    }
    
    /**
     * Creates an identity matrix.
     * 
     * @param n the size of the matrix (n x n)
     * @return an identity matrix
     */
    public static NDArray eye(int n) {
        return eye(n, n);
    }
    
    /**
     * Creates a 2D array with ones on the diagonal and zeros elsewhere.
     * 
     * @param n the number of rows
     * @param m the number of columns
     * @return a new NDArray
     */
    public static NDArray eye(int n, int m) {
        NDArray array = zeros(n, m);
        int minDim = Math.min(n, m);
        for (int i = 0; i < minDim; i++) {
            array.set(1.0, i, i);
        }
        return array;
    }
    
    /**
     * Creates an identity matrix (alias for eye).
     * 
     * @param n the size of the matrix
     * @return an identity matrix
     */
    public static NDArray identity(int n) {
        return eye(n);
    }
    
    /**
     * Extracts or constructs a diagonal array.
     * 
     * @param array the input array (must be 2D)
     * @param k the diagonal offset (0 = main diagonal, positive = upper, negative = lower)
     * @return a 1D array containing the diagonal
     */
    public static NDArray diag(NDArray array, int k) {
        if (array.getNdims() != 2) {
            throw new IllegalArgumentException("diag() requires a 2D array");
        }
        
        int[] shape = array.getShape();
        int rows = shape[0];
        int cols = shape[1];
        
        int diagLength;
        int startRow, startCol;
        
        if (k >= 0) {
            startRow = 0;
            startCol = k;
            diagLength = Math.min(rows, cols - k);
        } else {
            startRow = -k;
            startCol = 0;
            diagLength = Math.min(rows + k, cols);
        }
        
        if (diagLength <= 0) {
            return new NDArray(0);
        }
        
        double[] data = new double[diagLength];
        for (int i = 0; i < diagLength; i++) {
            data[i] = array.get(startRow + i, startCol + i);
        }
        
        return new NDArray(data, diagLength);
    }
    
    /**
     * Extracts the main diagonal.
     * 
     * @param array the input array
     * @return a 1D array containing the diagonal
     */
    public static NDArray diag(NDArray array) {
        return diag(array, 0);
    }
    
    /**
     * Creates a triangular matrix.
     * 
     * @param n the size of the matrix
     * @param k the diagonal offset
     * @return a triangular matrix
     */
    public static NDArray tri(int n, int k) {
        return tri(n, n, k);
    }
    
    /**
     * Creates a triangular matrix.
     * 
     * @param n the number of rows
     * @param m the number of columns
     * @param k the diagonal offset
     * @return a triangular matrix
     */
    public static NDArray tri(int n, int m, int k) {
        NDArray array = zeros(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (j <= i + k) {
                    array.set(1.0, i, j);
                }
            }
        }
        return array;
    }
    
    /**
     * Creates a lower triangular matrix.
     * 
     * @param array the input array
     * @param k the diagonal offset
     * @return a lower triangular matrix
     */
    public static NDArray tril(NDArray array, int k) {
        if (array.getNdims() != 2) {
            throw new IllegalArgumentException("tril() requires a 2D array");
        }
        
        NDArray result = new NDArray(array);
        int[] shape = result.getShape();
        int rows = shape[0];
        int cols = shape[1];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (j > i + k) {
                    result.set(0.0, i, j);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Creates a lower triangular matrix (main diagonal).
     * 
     * @param array the input array
     * @return a lower triangular matrix
     */
    public static NDArray tril(NDArray array) {
        return tril(array, 0);
    }
    
    /**
     * Creates an upper triangular matrix.
     * 
     * @param array the input array
     * @param k the diagonal offset
     * @return an upper triangular matrix
     */
    public static NDArray triu(NDArray array, int k) {
        if (array.getNdims() != 2) {
            throw new IllegalArgumentException("triu() requires a 2D array");
        }
        
        NDArray result = new NDArray(array);
        int[] shape = result.getShape();
        int rows = shape[0];
        int cols = shape[1];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (j < i + k) {
                    result.set(0.0, i, j);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Creates an upper triangular matrix (main diagonal).
     * 
     * @param array the input array
     * @return an upper triangular matrix
     */
    public static NDArray triu(NDArray array) {
        return triu(array, 0);
    }
    
    /**
     * Creates an array from a function applied to each index.
     * 
     * @param shape the shape of the array
     * @param function the function to apply
     * @return a new NDArray
     */
    public static NDArray fromFunction(int[] shape, DoubleFunction<Double> function) {
        NDArray array = new NDArray(shape);
        fillFromFunction(array, shape, new int[shape.length], 0, function);
        return array;
    }
    
    /**
     * Helper method to recursively fill array from function.
     */
    private static void fillFromFunction(NDArray array, int[] shape, int[] indices, int dim, DoubleFunction<Double> function) {
        if (dim == shape.length) {
            double value = function.apply(calculateIndexValue(indices));
            array.set(value, indices);
            return;
        }
        
        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            fillFromFunction(array, shape, indices, dim + 1, function);
        }
    }
    
    /**
     * Calculates a single value from multi-dimensional indices.
     */
    private static double calculateIndexValue(int[] indices) {
        // Simple linear combination for now
        double value = 0;
        for (int i = 0; i < indices.length; i++) {
            value += indices[i] * Math.pow(10, indices.length - i - 1);
        }
        return value;
    }
    
}

