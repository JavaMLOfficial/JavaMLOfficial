package com.javaml.dataframe;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Advanced DataFrame operations including groupby, merge, and aggregation.
 * Equivalent to Pandas groupby, merge, and aggregation functions.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class DataFrameOperations {
    
    private DataFrameOperations() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Groups a DataFrame by one or more columns.
     * 
     * @param df the DataFrame
     * @param columnNames the column names to group by
     * @return a DataFrameGroupBy object
     */
    public static DataFrameGroupBy groupBy(DataFrame df, String... columnNames) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        if (columnNames == null || columnNames.length == 0) {
            throw new IllegalArgumentException("At least one column name must be specified");
        }
        
        return new DataFrameGroupBy(df, Arrays.asList(columnNames));
    }
    
    /**
     * Merges two DataFrames (SQL-like join).
     * 
     * @param left the left DataFrame
     * @param right the right DataFrame
     * @param on the column name(s) to join on
     * @param how the join type (inner, outer, left, right)
     * @return a merged DataFrame
     */
    public static DataFrame merge(DataFrame left, DataFrame right, String on, String how) {
        return merge(left, right, new String[]{on}, how);
    }
    
    /**
     * Merges two DataFrames on multiple columns.
     * 
     * @param left the left DataFrame
     * @param right the right DataFrame
     * @param on the column name(s) to join on
     * @param how the join type (inner, outer, left, right)
     * @return a merged DataFrame
     */
    public static DataFrame merge(DataFrame left, DataFrame right, String[] on, String how) {
        if (left == null || right == null) {
            throw new IllegalArgumentException("DataFrames cannot be null");
        }
        if (on == null || on.length == 0) {
            throw new IllegalArgumentException("Join columns must be specified");
        }
        
        // Simplified merge implementation for single column
        if (on.length == 1) {
            return mergeSingleColumn(left, right, on[0], how);
        }
        
        throw new UnsupportedOperationException(
            "Multi-column merge not yet implemented");
    }
    
    /**
     * Merges on a single column.
     */
    private static DataFrame mergeSingleColumn(DataFrame left, DataFrame right, 
                                                String on, String how) {
        if (!left.getColumnNames().contains(on) || 
            !right.getColumnNames().contains(on)) {
            throw new IllegalArgumentException(
                "Join column must exist in both DataFrames");
        }
        
        Series leftKey = left.getColumn(on);
        Series rightKey = right.getColumn(on);
        
        // Build index maps
        Map<Object, List<Integer>> leftMap = new HashMap<>();
        Map<Object, List<Integer>> rightMap = new HashMap<>();
        
        for (int i = 0; i < left.getRowCount(); i++) {
            Object key = leftKey.getIndex().get(i);
            leftMap.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        
        for (int i = 0; i < right.getRowCount(); i++) {
            Object key = rightKey.getIndex().get(i);
            rightMap.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        
        // Determine which keys to include based on join type
        Set<Object> keys = new HashSet<>();
        switch (how.toLowerCase()) {
            case "inner":
                keys.addAll(leftMap.keySet());
                keys.retainAll(rightMap.keySet());
                break;
            case "outer":
                keys.addAll(leftMap.keySet());
                keys.addAll(rightMap.keySet());
                break;
            case "left":
                keys.addAll(leftMap.keySet());
                break;
            case "right":
                keys.addAll(rightMap.keySet());
                break;
            default:
                throw new IllegalArgumentException("Invalid join type: " + how);
        }
        
        // Build merged DataFrame
        List<Object> newRowLabels = new ArrayList<>();
        Map<String, List<Double>> newColumns = new LinkedHashMap<>();
        
        // Initialize columns
        for (String col : left.getColumnNames()) {
            if (!col.equals(on)) {
                newColumns.put(col, new ArrayList<>());
            }
        }
        for (String col : right.getColumnNames()) {
            if (!col.equals(on)) {
                String newColName = col;
                if (left.getColumnNames().contains(col)) {
                    newColName = col + "_y";
                }
                newColumns.put(newColName, new ArrayList<>());
            }
        }
        newColumns.put(on, new ArrayList<>());
        
        // Perform join
        for (Object key : keys) {
            List<Integer> leftIndices = leftMap.getOrDefault(key, Collections.emptyList());
            List<Integer> rightIndices = rightMap.getOrDefault(key, Collections.emptyList());
            
            if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
                // Handle outer join cases
                for (int leftIdx : leftIndices.isEmpty() ? 
                    Collections.singletonList(-1) : leftIndices) {
                    for (int rightIdx : rightIndices.isEmpty() ? 
                        Collections.singletonList(-1) : rightIndices) {
                        
                        newRowLabels.add(newRowLabels.size());
                        newColumns.get(on).add(key instanceof Number ? 
                            ((Number) key).doubleValue() : Double.NaN);
                        
                        // Add left columns
                        for (String col : left.getColumnNames()) {
                            if (!col.equals(on)) {
                                double value = leftIdx >= 0 ? 
                                    left.getColumn(col).get(leftIdx) : Double.NaN;
                                newColumns.get(col).add(value);
                            }
                        }
                        
                        // Add right columns
                        for (String col : right.getColumnNames()) {
                            if (!col.equals(on)) {
                                String newColName = col;
                                if (left.getColumnNames().contains(col)) {
                                    newColName = col + "_y";
                                }
                                double value = rightIdx >= 0 ? 
                                    right.getColumn(col).get(rightIdx) : Double.NaN;
                                newColumns.get(newColName).add(value);
                            }
                        }
                    }
                }
            } else {
                // Cartesian product for matching keys
                for (int leftIdx : leftIndices) {
                    for (int rightIdx : rightIndices) {
                        newRowLabels.add(newRowLabels.size());
                        newColumns.get(on).add(key instanceof Number ? 
                            ((Number) key).doubleValue() : Double.NaN);
                        
                        // Add left columns
                        for (String col : left.getColumnNames()) {
                            if (!col.equals(on)) {
                                newColumns.get(col).add(left.getColumn(col).get(leftIdx));
                            }
                        }
                        
                        // Add right columns
                        for (String col : right.getColumnNames()) {
                            if (!col.equals(on)) {
                                String newColName = col;
                                if (left.getColumnNames().contains(col)) {
                                    newColName = col + "_y";
                                }
                                newColumns.get(newColName).add(right.getColumn(col).get(rightIdx));
                            }
                        }
                    }
                }
            }
        }
        
        // Build result DataFrame
        Map<String, Series> resultData = new LinkedHashMap<>();
        Index newIndex = new Index(newRowLabels);
        
        for (Map.Entry<String, List<Double>> entry : newColumns.entrySet()) {
            double[] columnData = new double[entry.getValue().size()];
            for (int i = 0; i < entry.getValue().size(); i++) {
                columnData[i] = entry.getValue().get(i);
            }
            NDArray columnArray = new NDArray(columnData, columnData.length);
            Series series = new Series(columnArray, newIndex, entry.getKey());
            resultData.put(entry.getKey(), series);
        }
        
        return new DataFrame(resultData);
    }
    
    /**
     * Concatenates multiple DataFrames.
     * 
     * @param dataFrames the DataFrames to concatenate
     * @param axis the axis (0 for rows, 1 for columns)
     * @return a concatenated DataFrame
     */
    public static DataFrame concat(DataFrame[] dataFrames, int axis) {
        if (dataFrames == null || dataFrames.length == 0) {
            throw new IllegalArgumentException("DataFrames cannot be null or empty");
        }
        if (dataFrames.length == 1) {
            return dataFrames[0];
        }
        
        if (axis == 0) {
            return concatRows(dataFrames);
        } else if (axis == 1) {
            return concatColumns(dataFrames);
        } else {
            throw new IllegalArgumentException("Axis must be 0 or 1");
        }
    }
    
    /**
     * Concatenates DataFrames along rows (axis=0).
     */
    private static DataFrame concatRows(DataFrame[] dataFrames) {
        // Get all unique column names
        Set<String> allColumns = new LinkedHashSet<>();
        for (DataFrame df : dataFrames) {
            allColumns.addAll(df.getColumnNames());
        }
        
        Map<String, List<Double>> newColumns = new LinkedHashMap<>();
        List<Object> newRowLabels = new ArrayList<>();
        
        for (String col : allColumns) {
            newColumns.put(col, new ArrayList<>());
        }
        
        int rowOffset = 0;
        for (DataFrame df : dataFrames) {
            for (int i = 0; i < df.getRowCount(); i++) {
                newRowLabels.add(rowOffset + i);
            }
            
            for (String col : allColumns) {
                if (df.getColumnNames().contains(col)) {
                    for (int i = 0; i < df.getRowCount(); i++) {
                        newColumns.get(col).add(df.getColumn(col).get(i));
                    }
                } else {
                    // Fill with NaN for missing columns
                    for (int i = 0; i < df.getRowCount(); i++) {
                        newColumns.get(col).add(Double.NaN);
                    }
                }
            }
            rowOffset += df.getRowCount();
        }
        
        // Build result DataFrame
        Map<String, Series> resultData = new LinkedHashMap<>();
        Index newIndex = new Index(newRowLabels);
        
        for (Map.Entry<String, List<Double>> entry : newColumns.entrySet()) {
            double[] columnData = new double[entry.getValue().size()];
            for (int i = 0; i < entry.getValue().size(); i++) {
                columnData[i] = entry.getValue().get(i);
            }
            NDArray columnArray = new NDArray(columnData, columnData.length);
            Series series = new Series(columnArray, newIndex, entry.getKey());
            resultData.put(entry.getKey(), series);
        }
        
        return new DataFrame(resultData);
    }
    
    /**
     * Concatenates DataFrames along columns (axis=1).
     */
    private static DataFrame concatColumns(DataFrame[] dataFrames) {
        // All DataFrames must have the same number of rows
        int rowCount = dataFrames[0].getRowCount();
        for (DataFrame df : dataFrames) {
            if (df.getRowCount() != rowCount) {
                throw new IllegalArgumentException(
                    "All DataFrames must have the same number of rows for axis=1 concatenation");
            }
        }
        
        Map<String, Series> resultData = new LinkedHashMap<>();
        Index index = dataFrames[0].getIndex();
        
        for (DataFrame df : dataFrames) {
            for (String col : df.getColumnNames()) {
                String newColName = col;
                int suffix = 1;
                while (resultData.containsKey(newColName)) {
                    newColName = col + "_" + suffix;
                    suffix++;
                }
                resultData.put(newColName, df.getColumn(col));
            }
        }
        
        return new DataFrame(resultData);
    }
}

