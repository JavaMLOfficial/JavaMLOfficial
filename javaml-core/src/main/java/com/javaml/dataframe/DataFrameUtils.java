package com.javaml.dataframe;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;

import java.util.*;

/**
 * Utility methods for DataFrame operations.
 * Provides additional data manipulation methods.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class DataFrameUtils {
    
    private DataFrameUtils() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Drops rows with missing values.
     * 
     * @param df the DataFrame
     * @return a new DataFrame with missing values dropped
     */
    public static DataFrame dropna(DataFrame df) {
        return dropna(df, "any", null);
    }
    
    /**
     * Drops rows with missing values.
     * 
     * @param df the DataFrame
     * @param how "any" to drop if any NaN, "all" to drop if all NaN
     * @param subset columns to consider (null for all columns)
     * @return a new DataFrame with missing values dropped
     */
    public static DataFrame dropna(DataFrame df, String how, String[] subset) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        
        List<String> columnsToCheck = subset != null ? 
            Arrays.asList(subset) : df.getColumnNames();
        
        List<Integer> validRows = new ArrayList<>();
        
        for (int i = 0; i < df.getRowCount(); i++) {
            int nanCount = 0;
            for (String col : columnsToCheck) {
                double value = df.getColumn(col).get(i);
                if (Double.isNaN(value)) {
                    nanCount++;
                }
            }
            
            boolean keepRow = false;
            if ("any".equals(how)) {
                keepRow = nanCount == 0;
            } else if ("all".equals(how)) {
                keepRow = nanCount < columnsToCheck.size();
            }
            
            if (keepRow) {
                validRows.add(i);
            }
        }
        
        // Create new DataFrame with valid rows
        return df.iloc(validRows.stream().mapToInt(i -> i).toArray(), 
                      df.getColumnNames().stream().mapToInt(
                          col -> df.getColumnNames().indexOf(col)).toArray());
    }
    
    /**
     * Fills missing values.
     * 
     * @param df the DataFrame
     * @param value the value to fill with
     * @return a new DataFrame with missing values filled
     */
    public static DataFrame fillna(DataFrame df, double value) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        
        Map<String, Series> newData = new LinkedHashMap<>();
        
        for (String colName : df.getColumnNames()) {
            Series series = df.getColumn(colName);
            NDArray data = series.getData();
            double[] newDataArray = data.getData();
            
            for (int i = 0; i < newDataArray.length; i++) {
                if (Double.isNaN(newDataArray[i])) {
                    newDataArray[i] = value;
                }
            }
            
            NDArray newArray = new NDArray(newDataArray, newDataArray.length);
            Series newSeries = new Series(newArray, series.getIndex(), colName);
            newData.put(colName, newSeries);
        }
        
        return new DataFrame(newData);
    }
    
    /**
     * Drops duplicate rows.
     * 
     * @param df the DataFrame
     * @return a new DataFrame with duplicates removed
     */
    public static DataFrame dropDuplicates(DataFrame df) {
        return dropDuplicates(df, null, true);
    }
    
    /**
     * Drops duplicate rows.
     * 
     * @param df the DataFrame
     * @param subset columns to consider (null for all columns)
     * @param keepFirst if true, keep first occurrence; if false, keep last
     * @return a new DataFrame with duplicates removed
     */
    public static DataFrame dropDuplicates(DataFrame df, String[] subset, boolean keepFirst) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        
        List<String> columnsToCheck = subset != null ? 
            Arrays.asList(subset) : df.getColumnNames();
        
        Set<String> seen = new LinkedHashSet<>();
        List<Integer> keepRows = new ArrayList<>();
        
        for (int i = 0; i < df.getRowCount(); i++) {
            StringBuilder key = new StringBuilder();
            for (String col : columnsToCheck) {
                key.append(df.getColumn(col).get(i)).append("|");
            }
            String rowKey = key.toString();
            
            if (!seen.contains(rowKey)) {
                seen.add(rowKey);
                keepRows.add(i);
            } else if (!keepFirst) {
                // Remove previous occurrence and add this one
                keepRows.remove(keepRows.size() - 1);
                keepRows.add(i);
            }
        }
        
        return df.iloc(keepRows.stream().mapToInt(i -> i).toArray(),
                      df.getColumnNames().stream().mapToInt(
                          col -> df.getColumnNames().indexOf(col)).toArray());
    }
    
    /**
     * Sorts DataFrame by values.
     * 
     * @param df the DataFrame
     * @param by the column name(s) to sort by
     * @param ascending if true, sort ascending; if false, descending
     * @return a sorted DataFrame
     */
    public static DataFrame sortValues(DataFrame df, String[] by, boolean ascending) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        if (by == null || by.length == 0) {
            throw new IllegalArgumentException("At least one column must be specified");
        }
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < df.getRowCount(); i++) {
            indices.add(i);
        }
        
        // Sort indices based on column values
        indices.sort((a, b) -> {
            for (String col : by) {
                double valA = df.getColumn(col).get(a);
                double valB = df.getColumn(col).get(b);
                int cmp = Double.compare(valA, valB);
                if (cmp != 0) {
                    return ascending ? cmp : -cmp;
                }
            }
            return 0;
        });
        
        return df.iloc(indices.stream().mapToInt(i -> i).toArray(),
                      df.getColumnNames().stream().mapToInt(
                          col -> df.getColumnNames().indexOf(col)).toArray());
    }
    
    /**
     * Sorts DataFrame by values (ascending).
     * 
     * @param df the DataFrame
     * @param by the column name(s) to sort by
     * @return a sorted DataFrame
     */
    public static DataFrame sortValues(DataFrame df, String... by) {
        return sortValues(df, by, true);
    }
    
    /**
     * Renames columns or index.
     * 
     * @param df the DataFrame
     * @param columns mapping of old names to new names (null to skip)
     * @return a renamed DataFrame
     */
    public static DataFrame rename(DataFrame df, Map<String, String> columns) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        
        if (columns == null || columns.isEmpty()) {
            return df;
        }
        
        Map<String, Series> newData = new LinkedHashMap<>();
        
        for (String oldName : df.getColumnNames()) {
            String newName = columns.getOrDefault(oldName, oldName);
            newData.put(newName, df.getColumn(oldName));
        }
        
        return new DataFrame(newData);
    }
}

