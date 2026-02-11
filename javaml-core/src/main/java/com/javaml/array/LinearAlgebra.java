package com.javaml.array;

import java.util.Arrays;

/**
 * Linear algebra operations for NDArray, equivalent to NumPy's linalg module.
 * Provides functions for matrix operations, decompositions, and solving linear systems.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class LinearAlgebra {
    
    private LinearAlgebra() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Computes the dot product of two arrays.
     * For 2D arrays, this is matrix multiplication.
     * 
     * @param a the first array
     * @param b the second array
     * @return the dot product result
     */
    public static NDArray dot(NDArray a, NDArray b) {
        validateArrays(a, b);
        
        int[] shapeA = a.getShape();
        int[] shapeB = b.getShape();
        
        // Handle 1D x 1D (vector dot product)
        if (shapeA.length == 1 && shapeB.length == 1) {
            if (shapeA[0] != shapeB[0]) {
                throw new IllegalArgumentException(
                    "Arrays must have the same length for dot product");
            }
            double result = 0.0;
            for (int i = 0; i < shapeA[0]; i++) {
                result += a.get(i) * b.get(i);
            }
            return ArrayCreation.array(result);
        }
        
        // Handle 2D x 2D (matrix multiplication)
        if (shapeA.length == 2 && shapeB.length == 2) {
            return matmul(a, b);
        }
        
        // Handle 2D x 1D (matrix-vector multiplication)
        if (shapeA.length == 2 && shapeB.length == 1) {
            if (shapeA[1] != shapeB[0]) {
                throw new IllegalArgumentException(
                    "Matrix columns must match vector length");
            }
            double[] result = new double[shapeA[0]];
            for (int i = 0; i < shapeA[0]; i++) {
                double sum = 0.0;
                for (int j = 0; j < shapeA[1]; j++) {
                    sum += a.get(i, j) * b.get(j);
                }
                result[i] = sum;
            }
            return new NDArray(result, result.length);
        }
        
        throw new UnsupportedOperationException(
            "dot() currently supports 1D/2D arrays only");
    }
    
    /**
     * Computes matrix multiplication of two 2D arrays.
     * 
     * @param a the first matrix
     * @param b the second matrix
     * @return the matrix product
     */
    public static NDArray matmul(NDArray a, NDArray b) {
        validateArrays(a, b);
        
        int[] shapeA = a.getShape();
        int[] shapeB = b.getShape();
        
        if (shapeA.length != 2 || shapeB.length != 2) {
            throw new IllegalArgumentException("matmul() requires 2D arrays");
        }
        
        if (shapeA[1] != shapeB[0]) {
            throw new IllegalArgumentException(
                String.format("Matrix dimensions incompatible: %s x %s",
                    Arrays.toString(shapeA), Arrays.toString(shapeB)));
        }
        
        int rows = shapeA[0];
        int cols = shapeB[1];
        int inner = shapeA[1];
        
        NDArray result = new NDArray(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < inner; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(sum, i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Computes the inner product of two arrays.
     * 
     * @param a the first array
     * @param b the second array
     * @return the inner product
     */
    public static double inner(NDArray a, NDArray b) {
        validateArrays(a, b);
        
        if (a.getSize() != b.getSize()) {
            throw new IllegalArgumentException("Arrays must have the same size");
        }
        
        double result = 0.0;
        double[] dataA = a.getData();
        double[] dataB = b.getData();
        
        for (int i = 0; i < dataA.length; i++) {
            result += dataA[i] * dataB[i];
        }
        
        return result;
    }
    
    /**
     * Computes the outer product of two arrays.
     * 
     * @param a the first array
     * @param b the second array
     * @return the outer product matrix
     */
    public static NDArray outer(NDArray a, NDArray b) {
        validateArrays(a, b);
        
        int sizeA = a.getSize();
        int sizeB = b.getSize();
        
        NDArray result = new NDArray(sizeA, sizeB);
        double[] dataA = a.getData();
        double[] dataB = b.getData();
        
        for (int i = 0; i < sizeA; i++) {
            for (int j = 0; j < sizeB; j++) {
                result.set(dataA[i] * dataB[j], i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Computes the determinant of a square matrix.
     * 
     * @param matrix the input matrix
     * @return the determinant
     */
    public static double det(NDArray matrix) {
        validateArray(matrix);
        
        int[] shape = matrix.getShape();
        if (shape.length != 2 || shape[0] != shape[1]) {
            throw new IllegalArgumentException("det() requires a square matrix");
        }
        
        int n = shape[0];
        
        // For 1x1 and 2x2 matrices, use direct formulas
        if (n == 1) {
            return matrix.get(0, 0);
        }
        
        if (n == 2) {
            return matrix.get(0, 0) * matrix.get(1, 1) - matrix.get(0, 1) * matrix.get(1, 0);
        }
        
        // For larger matrices, use LU decomposition (simplified)
        // Full implementation would use a proper LU decomposition algorithm
        return detLU(matrix);
    }
    
    /**
     * Computes determinant using LU decomposition (simplified implementation).
     */
    private static double detLU(NDArray matrix) {
        int n = matrix.getShape()[0];
        NDArray lu = new NDArray(matrix);
        
        double det = 1.0;
        int[] pivots = new int[n];
        Arrays.fill(pivots, -1);
        
        // Simplified LU decomposition
        for (int i = 0; i < n; i++) {
            double maxVal = 0.0;
            int maxRow = i;
            
            for (int k = i; k < n; k++) {
                double absVal = Math.abs(lu.get(k, i));
                if (absVal > maxVal) {
                    maxVal = absVal;
                    maxRow = k;
                }
            }
            
            if (maxVal == 0.0) {
                return 0.0; // Singular matrix
            }
            
            if (maxRow != i) {
                // Swap rows
                for (int j = 0; j < n; j++) {
                    double temp = lu.get(i, j);
                    lu.set(lu.get(maxRow, j), i, j);
                    lu.set(temp, maxRow, j);
                }
                det *= -1;
            }
            
            pivots[i] = maxRow;
            det *= lu.get(i, i);
            
            for (int k = i + 1; k < n; k++) {
                lu.set(lu.get(k, i) / lu.get(i, i), k, i);
                for (int j = i + 1; j < n; j++) {
                    lu.set(lu.get(k, j) - lu.get(k, i) * lu.get(i, j), k, j);
                }
            }
        }
        
        return det;
    }
    
    /**
     * Computes the trace of a matrix (sum of diagonal elements).
     * 
     * @param matrix the input matrix
     * @return the trace
     */
    public static double trace(NDArray matrix) {
        validateArray(matrix);
        
        int[] shape = matrix.getShape();
        if (shape.length < 2) {
            throw new IllegalArgumentException("trace() requires at least a 2D array");
        }
        
        int minDim = Math.min(shape[0], shape[1]);
        double trace = 0.0;
        
        for (int i = 0; i < minDim; i++) {
            trace += matrix.get(i, i);
        }
        
        return trace;
    }
    
    /**
     * Computes the matrix rank.
     * 
     * @param matrix the input matrix
     * @return the rank
     */
    public static int matrixRank(NDArray matrix) {
        validateArray(matrix);
        
        // Simplified implementation - full version would use SVD
        int[] shape = matrix.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("matrixRank() requires a 2D array");
        }
        
        // For now, return a simple estimate
        // Full implementation would use SVD and count non-zero singular values
        return Math.min(shape[0], shape[1]);
    }
    
    /**
     * Validates that arrays are not null.
     */
    private static void validateArrays(NDArray a, NDArray b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Arrays cannot be null");
        }
    }
    
    /**
     * Computes vector dot product.
     * 
     * @param a the first vector
     * @param b the second vector
     * @return the dot product
     */
    public static double vdot(NDArray a, NDArray b) {
        validateArrays(a, b);
        
        if (a.getNdims() != 1 || b.getNdims() != 1) {
            throw new IllegalArgumentException("vdot() requires 1D arrays");
        }
        
        return inner(a, b);
    }
    
    /**
     * Computes matrix/vector norm.
     * 
     * @param array the input array
     * @param ord the order of the norm (1, 2, or -1 for inf)
     * @return the norm
     */
    public static double norm(NDArray array, int ord) {
        validateArray(array);
        
        if (array.getNdims() == 1) {
            // Vector norm
            double[] data = array.getData();
            if (ord == 1) {
                double sum = 0.0;
                for (double value : data) {
                    sum += Math.abs(value);
                }
                return sum;
            } else if (ord == 2) {
                double sum = 0.0;
                for (double value : data) {
                    sum += value * value;
                }
                return Math.sqrt(sum);
            } else if (ord == -1) {
                double max = 0.0;
                for (double value : data) {
                    max = Math.max(max, Math.abs(value));
                }
                return max;
            }
        } else if (array.getNdims() == 2) {
            // Matrix norm (Frobenius for ord=2)
            if (ord == 2) {
                double sum = 0.0;
                int[] shape = array.getShape();
                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        double value = array.get(i, j);
                        sum += value * value;
                    }
                }
                return Math.sqrt(sum);
            }
        }
        
        throw new UnsupportedOperationException(
            "norm() with ord=" + ord + " not yet fully implemented");
    }
    
    /**
     * Computes matrix/vector norm (default: Frobenius/2-norm).
     * 
     * @param array the input array
     * @return the norm
     */
    public static double norm(NDArray array) {
        return norm(array, 2);
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

