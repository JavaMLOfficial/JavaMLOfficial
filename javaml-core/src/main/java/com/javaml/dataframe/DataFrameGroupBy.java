package com.javaml.dataframe;

import com.javaml.array.ArrayStats;
import com.javaml.array.NDArray;

import java.util.*;
import java.util.stream.Collectors;

/**
 * GroupBy object for DataFrame, equivalent to Pandas GroupBy.
 * Provides aggregation and transformation operations on grouped data.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class DataFrameGroupBy {
    
    private final DataFrame df;
    private final List<String> groupByColumns;
    private final Map<List<Object>, List<Integer>> groups;
    
    /**
     * Creates a new DataFrameGroupBy object.
     * 
     * @param df the DataFrame to group
     * @param groupByColumns the columns to group by
     */
    DataFrameGroupBy(DataFrame df, List<String> groupByColumns) {
        this.df = df;
        this.groupByColumns = groupByColumns;
        this.groups = computeGroups();
    }
    
    /**
     * Computes the groups.
     */
    private Map<List<Object>, List<Integer>> computeGroups() {
        Map<List<Object>, List<Integer>> groups = new LinkedHashMap<>();
        
        for (int i = 0; i < df.getRowCount(); i++) {
            List<Object> key = new ArrayList<>();
            for (String col : groupByColumns) {
                Series series = df.getColumn(col);
                key.add(series.getIndex().get(i));
            }
            
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        
        return groups;
    }
    
    /**
     * Aggregates grouped data using the specified function.
     * 
     * @param column the column to aggregate
     * @param function the aggregation function (mean, sum, count, etc.)
     * @return a new DataFrame with aggregated results
     */
    public DataFrame agg(String column, String function) {
        if (!df.getColumnNames().contains(column)) {
            throw new IllegalArgumentException("Column not found: " + column);
        }
        
        Series series = df.getColumn(column);
        List<Object> groupKeys = new ArrayList<>();
        List<Double> aggregatedValues = new ArrayList<>();
        
        for (Map.Entry<List<Object>, List<Integer>> entry : groups.entrySet()) {
            groupKeys.add(entry.getKey());
            List<Integer> indices = entry.getValue();
            
            double[] values = new double[indices.size()];
            for (int i = 0; i < indices.size(); i++) {
                values[i] = series.get(indices.get(i));
            }
            NDArray groupArray = new NDArray(values, values.length);
            
            double aggregated = aggregate(groupArray, function);
            aggregatedValues.add(aggregated);
        }
        
        // Build result DataFrame
        Map<String, Series> resultData = new LinkedHashMap<>();
        Index resultIndex = new Index(groupKeys);
        
        // Add group by columns
        for (int i = 0; i < groupByColumns.size(); i++) {
            String colName = groupByColumns.get(i);
            double[] columnData = new double[groupKeys.size()];
            for (int j = 0; j < groupKeys.size(); j++) {
                List<Object> key = (List<Object>) groupKeys.get(j);
                Object value = key.get(i);
                columnData[j] = value instanceof Number ? 
                    ((Number) value).doubleValue() : Double.NaN;
            }
            NDArray columnArray = new NDArray(columnData, columnData.length);
            Series resultSeries = new Series(columnArray, resultIndex, colName);
            resultData.put(colName, resultSeries);
        }
        
        // Add aggregated column
        double[] aggData = new double[aggregatedValues.size()];
        for (int i = 0; i < aggregatedValues.size(); i++) {
            aggData[i] = aggregatedValues.get(i);
        }
        NDArray aggArray = new NDArray(aggData, aggData.length);
        Series aggSeries = new Series(aggArray, resultIndex, column + "_" + function);
        resultData.put(column + "_" + function, aggSeries);
        
        return new DataFrame(resultData);
    }
    
    /**
     * Aggregates using mean.
     * 
     * @param column the column to aggregate
     * @return a new DataFrame
     */
    public DataFrame mean(String column) {
        return agg(column, "mean");
    }
    
    /**
     * Aggregates using sum.
     * 
     * @param column the column to aggregate
     * @return a new DataFrame
     */
    public DataFrame sum(String column) {
        return agg(column, "sum");
    }
    
    /**
     * Aggregates using count.
     * 
     * @param column the column to aggregate
     * @return a new DataFrame
     */
    public DataFrame count(String column) {
        return agg(column, "count");
    }
    
    /**
     * Aggregates using min.
     * 
     * @param column the column to aggregate
     * @return a new DataFrame
     */
    public DataFrame min(String column) {
        return agg(column, "min");
    }
    
    /**
     * Aggregates using max.
     * 
     * @param column the column to aggregate
     * @return a new DataFrame
     */
    public DataFrame max(String column) {
        return agg(column, "max");
    }
    
    /**
     * Performs aggregation on an array.
     */
    private double aggregate(NDArray array, String function) {
        switch (function.toLowerCase()) {
            case "mean":
                return ArrayStats.mean(array);
            case "sum":
                return ArrayStats.sum(array);
            case "count":
                return array.getSize();
            case "min":
                return ArrayStats.min(array);
            case "max":
                return ArrayStats.max(array);
            case "std":
                return ArrayStats.std(array);
            case "var":
                return ArrayStats.var(array);
            default:
                throw new IllegalArgumentException("Unknown aggregation function: " + function);
        }
    }
}

