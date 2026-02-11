package com.javaml.dataframe;

import com.javaml.array.NDArray;

import java.util.*;

/**
 * One-dimensional labeled array (Series), equivalent to Pandas Series.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class Series {
    
    private final NDArray data;
    private final Index index;
    private final String name;
    
    /**
     * Creates a new Series from data with a default integer index.
     * 
     * @param data the data array
     */
    public Series(NDArray data) {
        this(data, null, null);
    }
    
    /**
     * Creates a new Series from data with the specified index.
     * 
     * @param data the data array
     * @param index the index
     */
    public Series(NDArray data, Index index) {
        this(data, index, null);
    }
    
    /**
     * Creates a new Series from data with the specified index and name.
     * 
     * @param data the data array
     * @param index the index
     * @param name the name of the series
     */
    public Series(NDArray data, Index index, String name) {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        if (data.getNdims() != 1) {
            throw new IllegalArgumentException("Series data must be 1D");
        }
        
        this.data = data;
        
        if (index == null) {
            this.index = Index.range(data.getSize());
        } else {
            if (index.size() != data.getSize()) {
                throw new IllegalArgumentException(
                    "Index size must match data size");
            }
            this.index = index;
        }
        
        this.name = name;
    }
    
    /**
     * Creates a new Series from a map (label -> value).
     * 
     * @param map the map of labels to values
     * @return a new Series
     */
    public static Series fromMap(Map<Object, Double> map) {
        if (map == null || map.isEmpty()) {
            throw new IllegalArgumentException("Map cannot be null or empty");
        }
        
        List<Object> labels = new ArrayList<>(map.keySet());
        double[] values = new double[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            values[i] = map.get(labels.get(i));
        }
        
        NDArray data = new NDArray(values, values.length);
        Index index = new Index(labels);
        return new Series(data, index);
    }
    
    /**
     * Gets the value at the specified label.
     * 
     * @param label the label
     * @return the value
     */
    public double get(Object label) {
        int position = index.getPosition(label);
        if (position == -1) {
            throw new IllegalArgumentException("Label not found: " + label);
        }
        return data.get(position);
    }
    
    /**
     * Gets the value at the specified position.
     * 
     * @param position the position
     * @return the value
     */
    public double get(int position) {
        return data.get(position);
    }
    
    /**
     * Gets the underlying NDArray.
     * 
     * @return the data array
     */
    public NDArray getData() {
        return data;
    }
    
    /**
     * Gets the index.
     * 
     * @return the index
     */
    public Index getIndex() {
        return index;
    }
    
    /**
     * Gets the name of the series.
     * 
     * @return the name, or null if not set
     */
    public String getName() {
        return name;
    }
    
    /**
     * Gets the size of the series.
     * 
     * @return the size
     */
    public int size() {
        return data.getSize();
    }
    
    /**
     * Gets a sub-series with the specified labels.
     * 
     * @param labels the labels to include
     * @return a new Series
     */
    public Series loc(Object... labels) {
        List<Double> values = new ArrayList<>();
        List<Object> newLabels = new ArrayList<>();
        
        for (Object label : labels) {
            int position = index.getPosition(label);
            if (position != -1) {
                values.add(data.get(position));
                newLabels.add(label);
            }
        }
        
        double[] newData = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            newData[i] = values.get(i);
        }
        
        return new Series(new NDArray(newData, newData.length), 
                         new Index(newLabels), name);
    }
    
    /**
     * Gets a sub-series with the specified integer positions.
     * 
     * @param positions the positions to include
     * @return a new Series
     */
    public Series iloc(int... positions) {
        double[] newData = new double[positions.length];
        List<Object> newLabels = new ArrayList<>();
        
        for (int i = 0; i < positions.length; i++) {
            newData[i] = data.get(positions[i]);
            newLabels.add(index.get(positions[i]));
        }
        
        return new Series(new NDArray(newData, newData.length),
                         new Index(newLabels), name);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Series(");
        if (name != null) {
            sb.append("name='").append(name).append("', ");
        }
        sb.append("size=").append(size()).append(")\n");
        
        int displaySize = Math.min(10, size());
        for (int i = 0; i < displaySize; i++) {
            sb.append(index.get(i)).append("    ").append(data.get(i)).append("\n");
        }
        if (size() > displaySize) {
            sb.append("...\n");
        }
        
        return sb.toString();
    }
}

