package com.javaml.array;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 * Multi-dimensional array implementation equivalent to NumPy's ndarray.
 * Supports efficient storage, broadcasting, and vectorized operations.
 * 
 * <p>This class provides the foundation for all array operations in JavaML,
 * supporting n-dimensional arrays with efficient memory layout and operations
 * optimized for virtual threads.</p>
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class NDArray {
    
    /** The underlying data storage */
    private final double[] data;
    
    /** The shape of the array (dimensions) */
    private final int[] shape;
    
    /** The strides for each dimension */
    private final int[] strides;
    
    /** Total number of elements */
    private final int size;
    
    /** Number of dimensions */
    private final int ndim;
    
    /**
     * Creates a new NDArray with the specified shape.
     * 
     * @param shape the dimensions of the array
     * @throws IllegalArgumentException if shape is null or contains non-positive values
     */
    public NDArray(int... shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be null or empty");
        }
        
        this.shape = Arrays.copyOf(shape, shape.length);
        this.ndim = shape.length;
        
        // Calculate size
        int calculatedSize = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive, got: " + Arrays.toString(shape));
            }
            calculatedSize *= dim;
        }
        this.size = calculatedSize;
        
        // Initialize data array
        this.data = new double[size];
        
        // Calculate strides (row-major order)
        this.strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    
    /**
     * Creates a new NDArray from existing data with the specified shape.
     * 
     * @param data the data to wrap
     * @param shape the dimensions of the array
     * @throws IllegalArgumentException if data length doesn't match shape
     */
    public NDArray(double[] data, int... shape) {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be null or empty");
        }
        
        this.shape = Arrays.copyOf(shape, shape.length);
        this.ndim = shape.length;
        
        // Calculate size
        int calculatedSize = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive, got: " + Arrays.toString(shape));
            }
            calculatedSize *= dim;
        }
        this.size = calculatedSize;
        
        if (data.length != size) {
            throw new IllegalArgumentException(
                String.format("Data length (%d) does not match shape size (%d)", data.length, size)
            );
        }
        
        this.data = Arrays.copyOf(data, data.length);
        
        // Calculate strides (row-major order)
        this.strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    
    /**
     * Creates a new NDArray from existing data (creates a copy).
     * 
     * @param other the NDArray to copy
     */
    public NDArray(NDArray other) {
        if (other == null) {
            throw new IllegalArgumentException("Other array cannot be null");
        }
        this.data = Arrays.copyOf(other.data, other.data.length);
        this.shape = Arrays.copyOf(other.shape, other.shape.length);
        this.strides = Arrays.copyOf(other.strides, other.strides.length);
        this.size = other.size;
        this.ndim = other.ndim;
    }
    
    /**
     * Gets the value at the specified indices.
     * 
     * @param indices the indices for each dimension
     * @return the value at the specified position
     * @throws IllegalArgumentException if indices don't match dimensions
     */
    public double get(int... indices) {
        validateIndices(indices);
        int index = calculateFlatIndex(indices);
        return data[index];
    }
    
    /**
     * Sets the value at the specified indices.
     * 
     * @param value the value to set
     * @param indices the indices for each dimension
     * @throws IllegalArgumentException if indices don't match dimensions
     */
    public void set(double value, int... indices) {
        validateIndices(indices);
        int index = calculateFlatIndex(indices);
        data[index] = value;
    }
    
    /**
     * Gets the underlying data array (returns a copy for safety).
     * 
     * @return a copy of the data array
     */
    public double[] getData() {
        return Arrays.copyOf(data, data.length);
    }
    
    /**
     * Fills the array with the specified value.
     * 
     * @param value the value to fill with
     * @return this array for method chaining
     */
    public NDArray fill(double value) {
        Arrays.fill(data, value);
        return this;
    }
    
    /**
     * Gets the flat index for the given multi-dimensional indices.
     * Package-private for use by ArrayCreation utilities.
     * 
     * @param indices the multi-dimensional indices
     * @return the flat index
     */
    int getFlatIndex(int... indices) {
        validateIndices(indices);
        return calculateFlatIndex(indices);
    }
    
    /**
     * Sets a value at the flat index (for internal use).
     * Package-private for use by utility classes.
     * 
     * @param flatIndex the flat index
     * @param value the value to set
     */
    void setFlat(int flatIndex, double value) {
        if (flatIndex < 0 || flatIndex >= size) {
            throw new IndexOutOfBoundsException("Flat index out of bounds: " + flatIndex);
        }
        data[flatIndex] = value;
    }
    
    /**
     * Gets a value at the flat index (for internal use).
     * Package-private for use by utility classes.
     * 
     * @param flatIndex the flat index
     * @return the value at the flat index
     */
    double getFlat(int flatIndex) {
        if (flatIndex < 0 || flatIndex >= size) {
            throw new IndexOutOfBoundsException("Flat index out of bounds: " + flatIndex);
        }
        return data[flatIndex];
    }
    
    /**
     * Gets the shape of the array.
     * 
     * @return a copy of the shape array
     */
    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }
    
    /**
     * Gets the number of dimensions.
     * 
     * @return the number of dimensions
     */
    public int getNdims() {
        return ndim;
    }
    
    /**
     * Gets the total number of elements.
     * 
     * @return the total size
     */
    public int getSize() {
        return size;
    }
    
    /**
     * Gets the stride for each dimension.
     * 
     * @return a copy of the strides array
     */
    public int[] getStrides() {
        return Arrays.copyOf(strides, strides.length);
    }
    
    /**
     * Validates that the indices match the array dimensions.
     */
    private void validateIndices(int[] indices) {
        if (indices == null || indices.length != ndim) {
            throw new IllegalArgumentException(
                String.format("Expected %d indices, got %s", ndim, 
                    indices == null ? "null" : indices.length)
            );
        }
        for (int i = 0; i < ndim; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    String.format("Index %d out of bounds for dimension %d (size: %d)", 
                        indices[i], i, shape[i])
                );
            }
        }
    }
    
    /**
     * Calculates the flat index from multi-dimensional indices.
     */
    private int calculateFlatIndex(int[] indices) {
        int index = 0;
        for (int i = 0; i < ndim; i++) {
            index += indices[i] * strides[i];
        }
        return index;
    }
    
    /**
     * Reshapes the array to the new shape (returns a view if possible, copy otherwise).
     * 
     * @param newShape the new shape
     * @return a reshaped NDArray
     * @throws IllegalArgumentException if the new shape is incompatible
     */
    public NDArray reshape(int... newShape) {
        if (newShape == null || newShape.length == 0) {
            throw new IllegalArgumentException("New shape cannot be null or empty");
        }
        
        // Calculate new size
        int newSize = 1;
        int inferredDim = -1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (inferredDim != -1) {
                    throw new IllegalArgumentException("Only one dimension can be inferred (-1)");
                }
                inferredDim = i;
            } else if (newShape[i] <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive or -1");
            } else {
                newSize *= newShape[i];
            }
        }
        
        // Infer dimension if needed
        if (inferredDim != -1) {
            if (size % newSize != 0) {
                throw new IllegalArgumentException("Cannot infer dimension: size is not divisible");
            }
            newShape[inferredDim] = size / newSize;
            newSize = size;
        }
        
        if (newSize != size) {
            throw new IllegalArgumentException(
                String.format("Cannot reshape array of size %d into shape %s", 
                    size, Arrays.toString(newShape))
            );
        }
        
        // Create new array with same data but new shape
        NDArray result = new NDArray(data, newShape);
        return result;
    }
    
    /**
     * Flattens the array to 1D.
     * 
     * @return a 1D view of the array
     */
    public NDArray flatten() {
        return reshape(size);
    }
    
    /**
     * Transposes the array (swaps axes).
     * 
     * @return a transposed NDArray
     */
    public NDArray transpose() {
        if (ndim != 2) {
            throw new UnsupportedOperationException("transpose() currently only supports 2D arrays");
        }
        return transpose(1, 0);
    }
    
    /**
     * Transposes the array with the specified axis order.
     * 
     * @param axes the new axis order
     * @return a transposed NDArray
     */
    public NDArray transpose(int... axes) {
        if (axes == null || axes.length != ndim) {
            throw new IllegalArgumentException("Axes must match number of dimensions");
        }
        
        // Validate axes
        boolean[] used = new boolean[ndim];
        for (int axis : axes) {
            if (axis < 0 || axis >= ndim) {
                throw new IllegalArgumentException("Invalid axis: " + axis);
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Duplicate axis: " + axis);
            }
            used[axis] = true;
        }
        
        // Create new shape and strides
        int[] newShape = new int[ndim];
        int[] newStrides = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            newShape[i] = shape[axes[i]];
            newStrides[i] = strides[axes[i]];
        }
        
        // Create transposed array
        NDArray result = new NDArray(newShape);
        for (int i = 0; i < size; i++) {
            int[] oldIndices = calculateMultiIndex(i);
            int[] newIndices = new int[ndim];
            for (int j = 0; j < ndim; j++) {
                newIndices[j] = oldIndices[axes[j]];
            }
            int newFlatIndex = 0;
            for (int j = 0; j < ndim; j++) {
                newFlatIndex += newIndices[j] * newStrides[j];
            }
            result.data[newFlatIndex] = data[i];
        }
        
        return result;
    }
    
    /**
     * Calculates multi-dimensional indices from a flat index.
     */
    private int[] calculateMultiIndex(int flatIndex) {
        int[] indices = new int[ndim];
        int remaining = flatIndex;
        for (int i = 0; i < ndim; i++) {
            indices[i] = remaining / strides[i];
            remaining = remaining % strides[i];
        }
        return indices;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NDArray ndArray = (NDArray) obj;
        return size == ndArray.size &&
               ndim == ndArray.ndim &&
               Arrays.equals(shape, ndArray.shape) &&
               Arrays.equals(data, ndArray.data);
    }
    
    @Override
    public int hashCode() {
        int result = Objects.hash(size, ndim);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NDArray(shape=").append(Arrays.toString(shape));
        sb.append(", size=").append(size);
        sb.append(", ndim=").append(ndim);
        sb.append(")\n");
        
        // Print array contents (simplified for now)
        if (size <= 100) {
            sb.append(Arrays.toString(data));
        } else {
            sb.append("[First 50: ");
            for (int i = 0; i < Math.min(50, size); i++) {
                sb.append(data[i]).append(", ");
            }
            sb.append("... (total: ").append(size).append(" elements)]");
        }
        
        return sb.toString();
    }
}

