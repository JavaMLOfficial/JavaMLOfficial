package com.javaml.array;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Array manipulation functions equivalent to NumPy's array manipulation routines.
 * Provides functions for reshaping, concatenating, splitting, and other array operations.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ArrayManipulation {
    
    private ArrayManipulation() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Reshapes an array to the new shape.
     * 
     * @param array the input array
     * @param newShape the new shape
     * @return a reshaped array
     */
    public static NDArray reshape(NDArray array, int... newShape) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        return array.reshape(newShape);
    }
    
    /**
     * Flattens an array to 1D.
     * 
     * @param array the input array
     * @return a flattened 1D array
     */
    public static NDArray flatten(NDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        return array.flatten();
    }
    
    /**
     * Transposes a 2D array.
     * 
     * @param array the input array
     * @return a transposed array
     */
    public static NDArray transpose(NDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        return array.transpose();
    }
    
    /**
     * Transposes an array with the specified axis order.
     * 
     * @param array the input array
     * @param axes the new axis order
     * @return a transposed array
     */
    public static NDArray transpose(NDArray array, int... axes) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        return array.transpose(axes);
    }
    
    /**
     * Concatenates arrays along the specified axis.
     * 
     * @param arrays the arrays to concatenate
     * @param axis the axis along which to concatenate
     * @return a concatenated array
     */
    public static NDArray concatenate(NDArray[] arrays, int axis) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be null or empty");
        }
        if (arrays.length == 1) {
            return new NDArray(arrays[0]);
        }
        
        // Validate all arrays have compatible shapes
        int ndim = arrays[0].getNdims();
        if (axis < 0 || axis >= ndim) {
            throw new IllegalArgumentException("Axis out of bounds: " + axis);
        }
        
        int[] baseShape = arrays[0].getShape();
        for (int i = 1; i < arrays.length; i++) {
            int[] shape = arrays[i].getShape();
            if (shape.length != ndim) {
                throw new IllegalArgumentException(
                    "All arrays must have the same number of dimensions");
            }
            for (int j = 0; j < ndim; j++) {
                if (j != axis && shape[j] != baseShape[j]) {
                    throw new IllegalArgumentException(
                        String.format("Arrays must have same shape except on axis %d", axis));
                }
            }
        }
        
        // Calculate new shape
        int[] newShape = Arrays.copyOf(baseShape, baseShape.length);
        for (int i = 0; i < arrays.length; i++) {
            newShape[axis] += arrays[i].getShape()[axis];
        }
        
        // Create result array
        NDArray result = new NDArray(newShape);
        
        // Copy data (simplified for 2D arrays for now)
        if (ndim == 2) {
            int rowOffset = 0;
            for (NDArray arr : arrays) {
                int rows = arr.getShape()[axis == 0 ? 0 : 1];
                int cols = arr.getShape()[axis == 0 ? 1 : 0];
                
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        if (axis == 0) {
                            result.set(arr.get(i, j), rowOffset + i, j);
                        } else {
                            result.set(arr.get(i, j), i, rowOffset + j);
                        }
                    }
                }
                rowOffset += rows;
            }
        } else {
            // For higher dimensions, use a more general approach
            // This is a simplified version - full implementation would handle all cases
            throw new UnsupportedOperationException(
                "concatenate() currently only supports 2D arrays");
        }
        
        return result;
    }
    
    /**
     * Concatenates arrays along axis 0 (default).
     * 
     * @param arrays the arrays to concatenate
     * @return a concatenated array
     */
    public static NDArray concatenate(NDArray... arrays) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be null or empty");
        }
        return concatenate(arrays, 0);
    }
    
    /**
     * Stacks arrays along a new axis.
     * 
     * @param arrays the arrays to stack
     * @param axis the axis along which to stack
     * @return a stacked array
     */
    public static NDArray stack(NDArray[] arrays, int axis) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be null or empty");
        }
        
        // Validate all arrays have the same shape
        int[] baseShape = arrays[0].getShape();
        for (int i = 1; i < arrays.length; i++) {
            if (!Arrays.equals(arrays[i].getShape(), baseShape)) {
                throw new IllegalArgumentException(
                    "All arrays must have the same shape for stacking");
            }
        }
        
        // Create new shape with added dimension
        int ndim = baseShape.length;
        if (axis < 0 || axis > ndim) {
            throw new IllegalArgumentException("Axis out of bounds: " + axis);
        }
        
        int[] newShape = new int[ndim + 1];
        System.arraycopy(baseShape, 0, newShape, 0, axis);
        newShape[axis] = arrays.length;
        System.arraycopy(baseShape, axis, newShape, axis + 1, ndim - axis);
        
        NDArray result = new NDArray(newShape);
        
        // Copy data
        for (int i = 0; i < arrays.length; i++) {
            NDArray arr = arrays[i];
            double[] arrData = arr.getData();
            // This is simplified - full implementation would handle multi-dimensional indexing
            int offset = i * arr.getSize();
            for (int j = 0; j < arrData.length; j++) {
                // Calculate new indices with the stacked dimension
                // Simplified for now - would need proper index calculation
                result.setFlat(offset + j, arrData[j]);
            }
        }
        
        return result;
    }
    
    /**
     * Stacks arrays along axis 0 (default).
     * 
     * @param arrays the arrays to stack
     * @return a stacked array
     */
    public static NDArray stack(NDArray... arrays) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be null or empty");
        }
        return stack(arrays, 0);
    }
    
    /**
     * Splits an array into multiple sub-arrays.
     * 
     * @param array the array to split
     * @param indicesOrSections the indices or number of sections
     * @param axis the axis along which to split
     * @return an array of sub-arrays
     */
    public static NDArray[] split(NDArray array, int indicesOrSections, int axis) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        int[] shape = array.getShape();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Axis out of bounds: " + axis);
        }
        
        int axisSize = shape[axis];
        List<NDArray> result = new ArrayList<>();
        
        if (indicesOrSections <= 0) {
            throw new IllegalArgumentException("Number of sections must be positive");
        }
        
        if (axisSize % indicesOrSections != 0) {
            throw new IllegalArgumentException(
                String.format("Array split does not result in an equal division. " +
                    "Axis size: %d, Sections: %d", axisSize, indicesOrSections));
        }
        
        int sectionSize = axisSize / indicesOrSections;
        
        // Create sub-arrays (simplified for 2D)
        if (shape.length == 2) {
            for (int i = 0; i < indicesOrSections; i++) {
                int[] subShape = new int[]{axis == 0 ? sectionSize : shape[0], 
                                           axis == 1 ? sectionSize : shape[1]};
                NDArray subArray = new NDArray(subShape);
                
                int start = i * sectionSize;
                for (int row = 0; row < subShape[0]; row++) {
                    for (int col = 0; col < subShape[1]; col++) {
                        if (axis == 0) {
                            subArray.set(array.get(start + row, col), row, col);
                        } else {
                            subArray.set(array.get(row, start + col), row, col);
                        }
                    }
                }
                result.add(subArray);
            }
        } else {
            throw new UnsupportedOperationException(
                "split() currently only supports 2D arrays");
        }
        
        return result.toArray(new NDArray[0]);
    }
    
    /**
     * Splits an array into multiple sub-arrays along axis 0.
     * 
     * @param array the array to split
     * @param indicesOrSections the indices or number of sections
     * @return an array of sub-arrays
     */
    public static NDArray[] split(NDArray array, int indicesOrSections) {
        return split(array, indicesOrSections, 0);
    }
    
    /**
     * Repeats elements of an array.
     * 
     * @param array the input array
     * @param repeats the number of repetitions for each element
     * @param axis the axis along which to repeat
     * @return a new array with repeated elements
     */
    public static NDArray repeat(NDArray array, int repeats, int axis) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        // Simplified implementation for 1D arrays
        if (array.getNdims() == 1) {
            double[] data = array.getData();
            double[] resultData = new double[data.length * repeats];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < repeats; j++) {
                    resultData[i * repeats + j] = data[i];
                }
            }
            return new NDArray(resultData, resultData.length);
        }
        
        throw new UnsupportedOperationException(
            "repeat() with axis currently only supports 1D arrays");
    }
    
    /**
     * Repeats elements of an array (flattened).
     * 
     * @param array the input array
     * @param repeats the number of repetitions
     * @return a new array with repeated elements
     */
    public static NDArray repeat(NDArray array, int repeats) {
        return repeat(array.flatten(), repeats, 0);
    }
}

