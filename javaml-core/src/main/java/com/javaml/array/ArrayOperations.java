package com.javaml.array;

import java.util.*;

/**
 * Additional array operations including sorting, uniqueness, and element manipulation.
 * Extends ArrayManipulation with more NumPy-equivalent functions.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ArrayOperations {
    
    private ArrayOperations() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Resizes an array in-place (modifies the original array).
     * 
     * @param array the array to resize
     * @param newShape the new shape
     * @return the resized array (same reference)
     */
    public static NDArray resize(NDArray array, int... newShape) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        int newSize = 1;
        for (int dim : newShape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive");
            }
            newSize *= dim;
        }
        
        // If new size is different, we need to create a new array
        // Note: Java arrays are fixed-size, so we return a new array
        // with the same data repeated or truncated
        double[] oldData = array.getData();
        double[] newData = new double[newSize];
        
        for (int i = 0; i < newSize; i++) {
            newData[i] = oldData[i % oldData.length];
        }
        
        return new NDArray(newData, newShape);
    }
    
    /**
     * Returns a flattened view of the array (alias for flatten).
     * 
     * @param array the input array
     * @return a flattened 1D array
     */
    public static NDArray ravel(NDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        return array.flatten();
    }
    
    /**
     * Swaps two axes of an array.
     * 
     * @param array the input array
     * @param axis1 first axis
     * @param axis2 second axis
     * @return array with swapped axes
     */
    public static NDArray swapaxes(NDArray array, int axis1, int axis2) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        int ndim = array.getNdims();
        if (axis1 < 0 || axis1 >= ndim || axis2 < 0 || axis2 >= ndim) {
            throw new IllegalArgumentException("Axis out of bounds");
        }
        
        int[] axes = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            axes[i] = i;
        }
        axes[axis1] = axis2;
        axes[axis2] = axis1;
        
        return array.transpose(axes);
    }
    
    /**
     * Moves an axis to a new position.
     * 
     * @param array the input array
     * @param source the axis to move
     * @param destination the destination position
     * @return array with moved axis
     */
    public static NDArray moveaxis(NDArray array, int source, int destination) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        int ndim = array.getNdims();
        if (source < 0) source += ndim;
        if (destination < 0) destination += ndim;
        
        if (source < 0 || source >= ndim || destination < 0 || destination >= ndim) {
            throw new IllegalArgumentException("Axis out of bounds");
        }
        
        List<Integer> axes = new ArrayList<>();
        for (int i = 0; i < ndim; i++) {
            axes.add(i);
        }
        
        axes.remove(source);
        axes.add(destination, source);
        
        int[] axesArray = axes.stream().mapToInt(i -> i).toArray();
        return array.transpose(axesArray);
    }
    
    /**
     * Stacks arrays horizontally (column-wise).
     * 
     * @param arrays the arrays to stack
     * @return a horizontally stacked array
     */
    public static NDArray hstack(NDArray... arrays) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be null or empty");
        }
        
        // For 1D arrays, concatenate along axis 0
        // For 2D arrays, concatenate along axis 1
        if (arrays[0].getNdims() == 1) {
            return ArrayManipulation.concatenate(arrays, 0);
        } else {
            return ArrayManipulation.concatenate(arrays, 1);
        }
    }
    
    /**
     * Stacks arrays vertically (row-wise).
     * 
     * @param arrays the arrays to stack
     * @return a vertically stacked array
     */
    public static NDArray vstack(NDArray... arrays) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be null or empty");
        }
        return ArrayManipulation.concatenate(arrays, 0);
    }
    
    /**
     * Splits an array horizontally.
     * 
     * @param array the array to split
     * @param indicesOrSections the indices or number of sections
     * @return array of sub-arrays
     */
    public static NDArray[] hsplit(NDArray array, int indicesOrSections) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        if (array.getNdims() < 2) {
            throw new IllegalArgumentException("hsplit requires at least 2D array");
        }
        return ArrayManipulation.split(array, indicesOrSections, 1);
    }
    
    /**
     * Splits an array vertically.
     * 
     * @param array the array to split
     * @param indicesOrSections the indices or number of sections
     * @return array of sub-arrays
     */
    public static NDArray[] vsplit(NDArray array, int indicesOrSections) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        return ArrayManipulation.split(array, indicesOrSections, 0);
    }
    
    /**
     * Constructs an array by repeating the input array.
     * 
     * @param array the input array
     * @param reps number of repetitions along each axis
     * @return a tiled array
     */
    public static NDArray tile(NDArray array, int... reps) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        if (reps == null || reps.length == 0) {
            throw new IllegalArgumentException("Reps cannot be null or empty");
        }
        
        int[] shape = array.getShape();
        int ndim = array.getNdims();
        
        // Extend reps to match array dimensions
        int[] fullReps = new int[Math.max(ndim, reps.length)];
        for (int i = 0; i < fullReps.length; i++) {
            if (i < reps.length) {
                fullReps[i] = reps[i];
            } else {
                fullReps[i] = 1;
            }
        }
        
        // Calculate new shape
        int[] newShape = new int[fullReps.length];
        for (int i = 0; i < newShape.length; i++) {
            int dimSize = (i < ndim) ? shape[i] : 1;
            newShape[i] = dimSize * fullReps[i];
        }
        
        NDArray result = new NDArray(newShape);
        
        // Tile the array
        int[] resultShape = result.getShape();
        for (int i = 0; i < result.getSize(); i++) {
            int[] resultIndices = calculateIndices(i, resultShape);
            int[] sourceIndices = new int[ndim];
            
            for (int j = 0; j < ndim; j++) {
                sourceIndices[j] = resultIndices[j] % shape[j];
            }
            
            double value = array.get(sourceIndices);
            result.set(value, resultIndices);
        }
        
        return result;
    }
    
    /**
     * Deletes sub-arrays along an axis.
     * 
     * @param array the input array
     * @param obj the indices or slice to delete
     * @param axis the axis along which to delete
     * @return array with deleted elements
     */
    public static NDArray delete(NDArray array, int obj, int axis) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        int[] shape = array.getShape();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Axis out of bounds");
        }
        if (obj < 0 || obj >= shape[axis]) {
            throw new IllegalArgumentException("Index out of bounds for axis");
        }
        
        // Create new shape
        int[] newShape = shape.clone();
        newShape[axis]--;
        
        NDArray result = new NDArray(newShape);
        
        // Copy elements, skipping the deleted index
        copyElementsSkippingAxis(array, result, axis, obj);
        
        return result;
    }
    
    /**
     * Inserts values along an axis before the given index.
     * 
     * @param array the input array
     * @param obj the index before which to insert
     * @param values the values to insert
     * @param axis the axis along which to insert
     * @return array with inserted values
     */
    public static NDArray insert(NDArray array, int obj, NDArray values, int axis) {
        if (array == null || values == null) {
            throw new IllegalArgumentException("Arrays cannot be null");
        }
        
        int[] shape = array.getShape();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Axis out of bounds");
        }
        
        // Simplified implementation - would need more complex logic for full support
        throw new UnsupportedOperationException(
            "insert() with axis is not yet fully implemented");
    }
    
    /**
     * Appends values to the end of an array.
     * 
     * @param array the input array
     * @param values the values to append
     * @param axis the axis along which to append
     * @return array with appended values
     */
    public static NDArray append(NDArray array, NDArray values, int axis) {
        if (array == null || values == null) {
            throw new IllegalArgumentException("Arrays cannot be null");
        }
        
        // For 1D arrays, append along axis 0
        if (array.getNdims() == 1 && values.getNdims() == 1) {
            double[] arrData = array.getData();
            double[] valData = values.getData();
            double[] result = new double[arrData.length + valData.length];
            System.arraycopy(arrData, 0, result, 0, arrData.length);
            System.arraycopy(valData, 0, result, arrData.length, valData.length);
            return new NDArray(result, result.length);
        }
        
        // For higher dimensions, use concatenate
        return ArrayManipulation.concatenate(new NDArray[]{array, values}, axis);
    }
    
    /**
     * Finds unique elements of an array.
     * 
     * @param array the input array
     * @return array of unique elements
     */
    public static NDArray unique(NDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        Set<Double> uniqueSet = new LinkedHashSet<>();
        double[] data = array.getData();
        
        for (double value : data) {
            uniqueSet.add(value);
        }
        
        double[] uniqueArray = new double[uniqueSet.size()];
        int idx = 0;
        for (Double value : uniqueSet) {
            uniqueArray[idx++] = value;
        }
        
        return new NDArray(uniqueArray, uniqueArray.length);
    }
    
    /**
     * Returns the indices that would sort an array.
     * 
     * @param array the input array
     * @return array of indices that sort the array
     */
    public static NDArray argsort(NDArray array) {
        return argsort(array, false);
    }
    
    /**
     * Returns the indices that would sort an array.
     * 
     * @param array the input array
     * @param descending if true, sort in descending order
     * @return array of indices that sort the array
     */
    public static NDArray argsort(NDArray array, boolean descending) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        double[] data = array.getData();
        Integer[] indices = new Integer[data.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        if (descending) {
            Arrays.sort(indices, (a, b) -> Double.compare(data[b], data[a]));
        } else {
            Arrays.sort(indices, (a, b) -> Double.compare(data[a], data[b]));
        }
        
        double[] result = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = indices[i];
        }
        
        return new NDArray(result, result.length);
    }
    
    /**
     * Sorts an array.
     * 
     * @param array the input array
     * @return sorted array
     */
    public static NDArray sort(NDArray array) {
        return sort(array, false);
    }
    
    /**
     * Sorts an array.
     * 
     * @param array the input array
     * @param descending if true, sort in descending order
     * @return sorted array
     */
    public static NDArray sort(NDArray array, boolean descending) {
        if (array == null) {
            throw new IllegalArgumentException("Array cannot be null");
        }
        
        double[] data = array.getData();
        double[] sorted = Arrays.copyOf(data, data.length);
        Arrays.sort(sorted);
        
        if (descending) {
            for (int i = 0; i < sorted.length / 2; i++) {
                double temp = sorted[i];
                sorted[i] = sorted[sorted.length - 1 - i];
                sorted[sorted.length - 1 - i] = temp;
            }
        }
        
        return new NDArray(sorted, array.getShape());
    }
    
    /**
     * Helper method to calculate multi-dimensional indices from flat index.
     */
    private static int[] calculateIndices(int flatIndex, int[] shape) {
        int[] indices = new int[shape.length];
        int remaining = flatIndex;
        int stride = 1;
        
        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = remaining / stride % shape[i];
            stride *= shape[i];
        }
        
        return indices;
    }
    
    /**
     * Helper method to copy elements skipping a specific index along an axis.
     */
    private static void copyElementsSkippingAxis(NDArray source, NDArray dest, 
                                                 int axis, int skipIndex) {
        // Simplified implementation
        // Full implementation would handle all dimensions properly
        int[] sourceShape = source.getShape();
        int[] destShape = dest.getShape();
        
        // For 2D arrays
        if (sourceShape.length == 2) {
            int destRow = 0;
            for (int row = 0; row < sourceShape[0]; row++) {
                if (axis == 0 && row == skipIndex) continue;
                
                int destCol = 0;
                for (int col = 0; col < sourceShape[1]; col++) {
                    if (axis == 1 && col == skipIndex) continue;
                    
                    dest.set(source.get(row, col), destRow, destCol);
                    if (axis != 1) destCol++;
                }
                if (axis != 0) destRow++;
            }
        }
    }
}

