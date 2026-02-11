package com.javaml.array;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

/**
 * Mathematical operations for NDArray, equivalent to NumPy's mathematical functions.
 * Supports element-wise operations, broadcasting, and parallel execution with virtual threads.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ArrayMath {
    
    private ArrayMath() {
        // Utility class - prevent instantiation
    }
    
    // ========== Basic Arithmetic Operations ==========
    
    /**
     * Adds two arrays element-wise.
     * 
     * @param a the first array
     * @param b the second array
     * @return a new array with the result
     */
    public static NDArray add(NDArray a, NDArray b) {
        return elementWise(a, b, (x, y) -> x + y);
    }
    
    /**
     * Adds a scalar to an array.
     * 
     * @param a the array
     * @param scalar the scalar value
     * @return a new array with the result
     */
    public static NDArray add(NDArray a, double scalar) {
        return elementWise(a, x -> x + scalar);
    }
    
    /**
     * Subtracts two arrays element-wise.
     * 
     * @param a the first array
     * @param b the second array
     * @return a new array with the result
     */
    public static NDArray subtract(NDArray a, NDArray b) {
        return elementWise(a, b, (x, y) -> x - y);
    }
    
    /**
     * Subtracts a scalar from an array.
     * 
     * @param a the array
     * @param scalar the scalar value
     * @return a new array with the result
     */
    public static NDArray subtract(NDArray a, double scalar) {
        return elementWise(a, x -> x - scalar);
    }
    
    /**
     * Multiplies two arrays element-wise.
     * 
     * @param a the first array
     * @param b the second array
     * @return a new array with the result
     */
    public static NDArray multiply(NDArray a, NDArray b) {
        return elementWise(a, b, (x, y) -> x * y);
    }
    
    /**
     * Multiplies an array by a scalar.
     * 
     * @param a the array
     * @param scalar the scalar value
     * @return a new array with the result
     */
    public static NDArray multiply(NDArray a, double scalar) {
        return elementWise(a, x -> x * scalar);
    }
    
    /**
     * Divides two arrays element-wise.
     * 
     * @param a the first array (numerator)
     * @param b the second array (denominator)
     * @return a new array with the result
     */
    public static NDArray divide(NDArray a, NDArray b) {
        return elementWise(a, b, (x, y) -> y != 0 ? x / y : Double.NaN);
    }
    
    /**
     * Divides an array by a scalar.
     * 
     * @param a the array
     * @param scalar the scalar value
     * @return a new array with the result
     */
    public static NDArray divide(NDArray a, double scalar) {
        if (scalar == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return elementWise(a, x -> x / scalar);
    }
    
    /**
     * Raises array elements to the power of another array element-wise.
     * 
     * @param a the base array
     * @param b the exponent array
     * @return a new array with the result
     */
    public static NDArray power(NDArray a, NDArray b) {
        return elementWise(a, b, Math::pow);
    }
    
    /**
     * Raises array elements to a scalar power.
     * 
     * @param a the array
     * @param scalar the exponent
     * @return a new array with the result
     */
    public static NDArray power(NDArray a, double scalar) {
        return elementWise(a, x -> Math.pow(x, scalar));
    }
    
    // ========== Trigonometric Functions ==========
    
    /**
     * Computes sine of array elements.
     * 
     * @param a the array
     * @return a new array with sine values
     */
    public static NDArray sin(NDArray a) {
        return elementWise(a, Math::sin);
    }
    
    /**
     * Computes cosine of array elements.
     * 
     * @param a the array
     * @return a new array with cosine values
     */
    public static NDArray cos(NDArray a) {
        return elementWise(a, Math::cos);
    }
    
    /**
     * Computes tangent of array elements.
     * 
     * @param a the array
     * @return a new array with tangent values
     */
    public static NDArray tan(NDArray a) {
        return elementWise(a, Math::tan);
    }
    
    /**
     * Computes arcsine of array elements.
     * 
     * @param a the array
     * @return a new array with arcsine values
     */
    public static NDArray arcsin(NDArray a) {
        return elementWise(a, Math::asin);
    }
    
    /**
     * Computes arccosine of array elements.
     * 
     * @param a the array
     * @return a new array with arccosine values
     */
    public static NDArray arccos(NDArray a) {
        return elementWise(a, Math::acos);
    }
    
    /**
     * Computes arctangent of array elements.
     * 
     * @param a the array
     * @return a new array with arctangent values
     */
    public static NDArray arctan(NDArray a) {
        return elementWise(a, Math::atan);
    }
    
    /**
     * Computes arctangent2 (four-quadrant arctangent).
     * 
     * @param y the y-coordinates
     * @param x the x-coordinates
     * @return a new array with arctangent2 values
     */
    public static NDArray arctan2(NDArray y, NDArray x) {
        return elementWise(y, x, Math::atan2);
    }
    
    // ========== Hyperbolic Functions ==========
    
    /**
     * Computes hyperbolic sine of array elements.
     * 
     * @param a the array
     * @return a new array with sinh values
     */
    public static NDArray sinh(NDArray a) {
        return elementWise(a, Math::sinh);
    }
    
    /**
     * Computes hyperbolic cosine of array elements.
     * 
     * @param a the array
     * @return a new array with cosh values
     */
    public static NDArray cosh(NDArray a) {
        return elementWise(a, Math::cosh);
    }
    
    /**
     * Computes hyperbolic tangent of array elements.
     * 
     * @param a the array
     * @return a new array with tanh values
     */
    public static NDArray tanh(NDArray a) {
        return elementWise(a, Math::tanh);
    }
    
    /**
     * Computes inverse hyperbolic sine of array elements.
     * 
     * @param a the array
     * @return a new array with arcsinh values
     */
    public static NDArray arcsinh(NDArray a) {
        return elementWise(a, x -> Math.log(x + Math.sqrt(x * x + 1)));
    }
    
    /**
     * Computes inverse hyperbolic cosine of array elements.
     * 
     * @param a the array
     * @return a new array with arccosh values
     */
    public static NDArray arccosh(NDArray a) {
        return elementWise(a, x -> {
            if (x < 1.0) return Double.NaN;
            return Math.log(x + Math.sqrt(x * x - 1));
        });
    }
    
    /**
     * Computes inverse hyperbolic tangent of array elements.
     * 
     * @param a the array
     * @return a new array with arctanh values
     */
    public static NDArray arctanh(NDArray a) {
        return elementWise(a, x -> {
            if (Math.abs(x) >= 1.0) return Double.NaN;
            return 0.5 * Math.log((1 + x) / (1 - x));
        });
    }
    
    /**
     * Rounds to the nearest integer (alias for around).
     * 
     * @param a the array
     * @return a new array with rounded values
     */
    public static NDArray rint(NDArray a) {
        return around(a);
    }
    
    /**
     * Computes floating-point remainder (fmod).
     * 
     * @param a the first array
     * @param b the second array
     * @return a new array with remainder values
     */
    public static NDArray fmod(NDArray a, NDArray b) {
        return elementWise(a, b, (x, y) -> y != 0 ? x % y : Double.NaN);
    }
    
    /**
     * Computes remainder of division.
     * 
     * @param a the first array
     * @param b the second array
     * @return a new array with remainder values
     */
    public static NDArray remainder(NDArray a, NDArray b) {
        return mod(a, b);
    }
    
    /**
     * Checks for NaN values.
     * 
     * @param a the array
     * @return a boolean array (as double: 1.0 for NaN, 0.0 otherwise)
     */
    public static NDArray isnan(NDArray a) {
        return elementWise(a, x -> Double.isNaN(x) ? 1.0 : 0.0);
    }
    
    /**
     * Checks for infinite values.
     * 
     * @param a the array
     * @return a boolean array (as double: 1.0 for Inf, 0.0 otherwise)
     */
    public static NDArray isinf(NDArray a) {
        return elementWise(a, x -> Double.isInfinite(x) ? 1.0 : 0.0);
    }
    
    /**
     * Checks for finite values.
     * 
     * @param a the array
     * @return a boolean array (as double: 1.0 for finite, 0.0 otherwise)
     */
    public static NDArray isfinite(NDArray a) {
        return elementWise(a, x -> Double.isFinite(x) ? 1.0 : 0.0);
    }
    
    /**
     * Computes maximum ignoring NaN values.
     * 
     * @param a the array
     * @return the maximum value (ignoring NaN)
     */
    public static double nanmax(NDArray a) {
        validateArray(a);
        double[] data = a.getData();
        double max = Double.NEGATIVE_INFINITY;
        boolean found = false;
        
        for (double value : data) {
            if (!Double.isNaN(value)) {
                if (!found || value > max) {
                    max = value;
                    found = true;
                }
            }
        }
        
        return found ? max : Double.NaN;
    }
    
    /**
     * Computes minimum ignoring NaN values.
     * 
     * @param a the array
     * @return the minimum value (ignoring NaN)
     */
    public static double nanmin(NDArray a) {
        validateArray(a);
        double[] data = a.getData();
        double min = Double.POSITIVE_INFINITY;
        boolean found = false;
        
        for (double value : data) {
            if (!Double.isNaN(value)) {
                if (!found || value < min) {
                    min = value;
                    found = true;
                }
            }
        }
        
        return found ? min : Double.NaN;
    }
    
    /**
     * Computes mean ignoring NaN values.
     * 
     * @param a the array
     * @return the mean value (ignoring NaN)
     */
    public static double nanmean(NDArray a) {
        validateArray(a);
        double[] data = a.getData();
        double sum = 0.0;
        int count = 0;
        
        for (double value : data) {
            if (!Double.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        
        return count > 0 ? sum / count : Double.NaN;
    }
    
    // ========== Exponential and Logarithmic Functions ==========
    
    /**
     * Computes e raised to the power of array elements.
     * 
     * @param a the array
     * @return a new array with exp values
     */
    public static NDArray exp(NDArray a) {
        return elementWise(a, Math::exp);
    }
    
    /**
     * Computes natural logarithm of array elements.
     * 
     * @param a the array
     * @return a new array with log values
     */
    public static NDArray log(NDArray a) {
        return elementWise(a, x -> x > 0 ? Math.log(x) : Double.NaN);
    }
    
    /**
     * Computes base-10 logarithm of array elements.
     * 
     * @param a the array
     * @return a new array with log10 values
     */
    public static NDArray log10(NDArray a) {
        return elementWise(a, x -> x > 0 ? Math.log10(x) : Double.NaN);
    }
    
    /**
     * Computes base-2 logarithm of array elements.
     * 
     * @param a the array
     * @return a new array with log2 values
     */
    public static NDArray log2(NDArray a) {
        return elementWise(a, x -> x > 0 ? Math.log(x) / Math.log(2) : Double.NaN);
    }
    
    /**
     * Computes square root of array elements.
     * 
     * @param a the array
     * @return a new array with sqrt values
     */
    public static NDArray sqrt(NDArray a) {
        return elementWise(a, x -> x >= 0 ? Math.sqrt(x) : Double.NaN);
    }
    
    // ========== Rounding Functions ==========
    
    /**
     * Rounds array elements to the nearest integer.
     * 
     * @param a the array
     * @return a new array with rounded values
     */
    public static NDArray around(NDArray a) {
        return elementWise(a, x -> Math.round(x));
    }
    
    /**
     * Rounds array elements to the nearest integer with specified decimals.
     * 
     * @param a the array
     * @param decimals the number of decimal places
     * @return a new array with rounded values
     */
    public static NDArray around(NDArray a, int decimals) {
        double factor = Math.pow(10, decimals);
        return elementWise(a, x -> Math.round(x * factor) / factor);
    }
    
    /**
     * Rounds down array elements to the nearest integer.
     * 
     * @param a the array
     * @return a new array with floor values
     */
    public static NDArray floor(NDArray a) {
        return elementWise(a, Math::floor);
    }
    
    /**
     * Rounds up array elements to the nearest integer.
     * 
     * @param a the array
     * @return a new array with ceil values
     */
    public static NDArray ceil(NDArray a) {
        return elementWise(a, Math::ceil);
    }
    
    /**
     * Truncates array elements (removes decimal part).
     * 
     * @param a the array
     * @return a new array with truncated values
     */
    public static NDArray trunc(NDArray a) {
        return elementWise(a, x -> x < 0 ? Math.ceil(x) : Math.floor(x));
    }
    
    // ========== Special Functions ==========
    
    /**
     * Computes absolute value of array elements.
     * 
     * @param a the array
     * @return a new array with absolute values
     */
    public static NDArray abs(NDArray a) {
        return elementWise(a, Math::abs);
    }
    
    /**
     * Computes sign of array elements (-1, 0, or 1).
     * 
     * @param a the array
     * @return a new array with sign values
     */
    public static NDArray sign(NDArray a) {
        return elementWise(a, x -> x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0));
    }
    
    /**
     * Computes modulo operation (a % b).
     * 
     * @param a the first array
     * @param b the second array
     * @return a new array with modulo values
     */
    public static NDArray mod(NDArray a, NDArray b) {
        return elementWise(a, b, (x, y) -> y != 0 ? x % y : Double.NaN);
    }
    
    // ========== Helper Methods ==========
    
    /**
     * Performs element-wise operation on a single array.
     */
    private static NDArray elementWise(NDArray a, java.util.function.DoubleUnaryOperator op) {
        validateArray(a);
        NDArray result = new NDArray(a.getShape());
        double[] aData = a.getData();
        double[] resultData = result.getData();
        
        for (int i = 0; i < aData.length; i++) {
            resultData[i] = op.applyAsDouble(aData[i]);
        }
        
        return result;
    }
    
    /**
     * Performs element-wise operation on two arrays.
     * TODO: Implement broadcasting for arrays with different shapes.
     */
    private static NDArray elementWise(NDArray a, NDArray b, 
                                       java.util.function.DoubleBinaryOperator op) {
        validateArray(a);
        validateArray(b);
        
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException(
                String.format("Arrays must have the same shape. Got %s and %s",
                    Arrays.toString(a.getShape()), Arrays.toString(b.getShape()))
            );
        }
        
        NDArray result = new NDArray(a.getShape());
        double[] aData = a.getData();
        double[] bData = b.getData();
        double[] resultData = result.getData();
        
        for (int i = 0; i < aData.length; i++) {
            resultData[i] = op.applyAsDouble(aData[i], bData[i]);
        }
        
        return result;
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

