package com.javaml.dataframe;

import java.util.*;

/**
 * Immutable index for labeling rows and columns in DataFrame and Series.
 * Equivalent to Pandas Index.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class Index {
    
    private final List<Object> labels;
    private final String name;
    
    /**
     * Creates a new Index with the specified labels.
     * 
     * @param labels the index labels
     */
    public Index(Object... labels) {
        this(null, labels);
    }
    
    /**
     * Creates a new Index with the specified name and labels.
     * 
     * @param name the name of the index
     * @param labels the index labels
     */
    public Index(String name, Object... labels) {
        if (labels == null || labels.length == 0) {
            throw new IllegalArgumentException("Labels cannot be null or empty");
        }
        this.labels = Collections.unmodifiableList(new ArrayList<>(Arrays.asList(labels)));
        this.name = name;
    }
    
    /**
     * Creates a new Index with the specified labels.
     * 
     * @param labels the index labels
     */
    public Index(List<Object> labels) {
        this(null, labels);
    }
    
    /**
     * Creates a new Index with the specified name and labels.
     * 
     * @param name the name of the index
     * @param labels the index labels
     */
    public Index(String name, List<Object> labels) {
        if (labels == null || labels.isEmpty()) {
            throw new IllegalArgumentException("Labels cannot be null or empty");
        }
        this.labels = Collections.unmodifiableList(new ArrayList<>(labels));
        this.name = name;
    }
    
    /**
     * Creates a RangeIndex (integer index from 0 to size-1).
     * 
     * @param size the size of the index
     * @return a new RangeIndex
     */
    public static Index range(int size) {
        return range(null, size);
    }
    
    /**
     * Creates a RangeIndex with a name.
     * 
     * @param name the name of the index
     * @param size the size of the index
     * @return a new RangeIndex
     */
    public static Index range(String name, int size) {
        List<Object> labels = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            labels.add(i);
        }
        return new Index(name, labels);
    }
    
    /**
     * Gets the label at the specified position.
     * 
     * @param position the position
     * @return the label
     */
    public Object get(int position) {
        if (position < 0 || position >= labels.size()) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + position);
        }
        return labels.get(position);
    }
    
    /**
     * Gets the position of the first occurrence of the label.
     * 
     * @param label the label to find
     * @return the position, or -1 if not found
     */
    public int getPosition(Object label) {
        return labels.indexOf(label);
    }
    
    /**
     * Gets all positions of the label (for duplicate labels).
     * 
     * @param label the label to find
     * @return a list of positions
     */
    public List<Integer> getPositions(Object label) {
        List<Integer> positions = new ArrayList<>();
        for (int i = 0; i < labels.size(); i++) {
            if (Objects.equals(labels.get(i), label)) {
                positions.add(i);
            }
        }
        return positions;
    }
    
    /**
     * Gets the size of the index.
     * 
     * @return the size
     */
    public int size() {
        return labels.size();
    }
    
    /**
     * Gets the name of the index.
     * 
     * @return the name, or null if not set
     */
    public String getName() {
        return name;
    }
    
    /**
     * Gets all labels as an unmodifiable list.
     * 
     * @return the labels
     */
    public List<Object> getLabels() {
        return labels;
    }
    
    /**
     * Checks if the index contains the label.
     * 
     * @param label the label to check
     * @return true if contained
     */
    public boolean contains(Object label) {
        return labels.contains(label);
    }
    
    /**
     * Checks if the index has duplicate labels.
     * 
     * @return true if duplicates exist
     */
    public boolean hasDuplicates() {
        Set<Object> seen = new HashSet<>();
        for (Object label : labels) {
            if (!seen.add(label)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Gets a sub-index with the specified positions.
     * 
     * @param positions the positions to include
     * @return a new Index
     */
    public Index subIndex(int... positions) {
        List<Object> subLabels = new ArrayList<>();
        for (int pos : positions) {
            subLabels.add(labels.get(pos));
        }
        return new Index(name, subLabels);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Index index = (Index) obj;
        return Objects.equals(labels, index.labels) && Objects.equals(name, index.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(labels, name);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Index(");
        if (name != null) {
            sb.append("name='").append(name).append("', ");
        }
        sb.append("size=").append(size()).append(")");
        if (size() <= 10) {
            sb.append("\n").append(labels);
        } else {
            sb.append("\n[").append(labels.subList(0, 5))
              .append(", ..., ").append(labels.subList(size() - 5, size())).append("]");
        }
        return sb.toString();
    }
}

