package com.javaml.dataframe;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;

import java.util.*;

/**
 * Statistical operations for DataFrame.
 * Provides methods like describe(), value_counts(), etc.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class DataFrameStats {
    
    private DataFrameStats() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Generates descriptive statistics for DataFrame columns.
     * 
     * @param df the DataFrame
     * @return a new DataFrame with summary statistics
     */
    public static DataFrame describe(DataFrame df) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        
        Map<String, Series> stats = new LinkedHashMap<>();
        List<String> statNames = Arrays.asList("count", "mean", "std", "min", 
                                                "25%", "50%", "75%", "max");
        
        for (String colName : df.getColumnNames()) {
            Series column = df.getColumn(colName);
            NDArray data = column.getData();
            
            double[] statValues = new double[statNames.size()];
            statValues[0] = data.getSize(); // count
            statValues[1] = ArrayStats.mean(data); // mean
            statValues[2] = ArrayStats.std(data); // std
            statValues[3] = ArrayStats.min(data); // min
            statValues[4] = ArrayStats.percentile(data, 25.0); // 25%
            statValues[5] = ArrayStats.median(data); // 50%
            statValues[6] = ArrayStats.percentile(data, 75.0); // 75%
            statValues[7] = ArrayStats.max(data); // max
            
            NDArray statArray = new NDArray(statValues, statValues.length);
            Series statSeries = new Series(statArray, Index.of(statNames), colName);
            stats.put(colName, statSeries);
        }
        
        return new DataFrame(stats);
    }
    
    /**
     * Counts unique values in a Series.
     * 
     * @param series the Series
     * @return a new Series with value counts
     */
    public static Series valueCounts(Series series) {
        if (series == null) {
            throw new IllegalArgumentException("Series cannot be null");
        }
        
        Map<Double, Integer> counts = new LinkedHashMap<>();
        NDArray data = series.getData();
        
        for (int i = 0; i < data.getSize(); i++) {
            double value = data.get(i);
            counts.put(value, counts.getOrDefault(value, 0) + 1);
        }
        
        // Sort by count (descending)
        List<Map.Entry<Double, Integer>> sorted = new ArrayList<>(counts.entrySet());
        sorted.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        double[] values = new double[sorted.size()];
        double[] countsArray = new double[sorted.size()];
        
        for (int i = 0; i < sorted.size(); i++) {
            values[i] = sorted.get(i).getKey();
            countsArray[i] = sorted.get(i).getValue();
        }
        
        NDArray valueArray = new NDArray(values, values.length);
        NDArray countArray = new NDArray(countsArray, countsArray.length);
        
        return new Series(countArray, Index.of(valueArray), series.getName() + "_count");
    }
    
    /**
     * Counts unique values in a DataFrame column.
     * 
     * @param df the DataFrame
     * @param column the column name
     * @return a new Series with value counts
     */
    public static Series valueCounts(DataFrame df, String column) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        return valueCounts(df.getColumn(column));
    }
    
    /**
     * Gets the number of unique values in a Series.
     * 
     * @param series the Series
     * @return the number of unique values
     */
    public static int nunique(Series series) {
        if (series == null) {
            throw new IllegalArgumentException("Series cannot be null");
        }
        
        Set<Double> unique = new HashSet<>();
        NDArray data = series.getData();
        
        for (int i = 0; i < data.getSize(); i++) {
            unique.add(data.get(i));
        }
        
        return unique.size();
    }
    
    /**
     * Gets the number of unique values in a DataFrame column.
     * 
     * @param df the DataFrame
     * @param column the column name
     * @return the number of unique values
     */
    public static int nunique(DataFrame df, String column) {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null");
        }
        return nunique(df.getColumn(column));
    }
    
    /**
     * Gets unique values in a Series.
     * 
     * @param series the Series
     * @return a new Series with unique values
     */
    public static Series unique(Series series) {
        if (series == null) {
            throw new IllegalArgumentException("Series cannot be null");
        }
        
        Set<Double> uniqueSet = new LinkedHashSet<>();
        NDArray data = series.getData();
        
        for (int i = 0; i < data.getSize(); i++) {
            uniqueSet.add(data.get(i));
        }
        
        double[] uniqueArray = new double[uniqueSet.size()];
        int idx = 0;
        for (Double value : uniqueSet) {
            uniqueArray[idx++] = value;
        }
        
        NDArray uniqueNDArray = new NDArray(uniqueArray, uniqueArray.length);
        return new Series(uniqueNDArray, Index.range(uniqueArray.length), 
                         series.getName() + "_unique");
    }
}

