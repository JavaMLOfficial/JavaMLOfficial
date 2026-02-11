package com.javaml.dataframe;

import com.javaml.array.NDArray;

import java.util.*;

/**
 * Two-dimensional labeled data structure (DataFrame), equivalent to Pandas DataFrame.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class DataFrame {
    
    private final Map<String, Series> columns;
    private final Index index;
    
    /**
     * Creates a new DataFrame from a map of column names to Series.
     * 
     * @param data the column data
     */
    public DataFrame(Map<String, Series> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }
        
        this.columns = new LinkedHashMap<>(data);
        
        // Validate all series have the same size
        int size = -1;
        for (Series series : columns.values()) {
            if (size == -1) {
                size = series.size();
            } else if (series.size() != size) {
                throw new IllegalArgumentException(
                    "All series must have the same size");
            }
        }
        
        // Use the index from the first series, or create a default one
        Series firstSeries = columns.values().iterator().next();
        this.index = firstSeries.getIndex();
    }
    
    /**
     * Creates a new DataFrame from an NDArray with column names.
     * 
     * @param data the 2D array
     * @param columns the column names
     */
    public DataFrame(NDArray data, String... columns) {
        this(data, Index.range(data.getShape()[0]), columns);
    }
    
    /**
     * Creates a new DataFrame from an NDArray with index and column names.
     * 
     * @param data the 2D array
     * @param index the row index
     * @param columns the column names
     */
    public DataFrame(NDArray data, Index index, String... columns) {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        if (data.getNdims() != 2) {
            throw new IllegalArgumentException("DataFrame data must be 2D");
        }
        if (columns == null || columns.length == 0) {
            throw new IllegalArgumentException("Columns cannot be null or empty");
        }
        
        int[] shape = data.getShape();
        int rows = shape[0];
        int cols = shape[1];
        
        if (columns.length != cols) {
            throw new IllegalArgumentException(
                "Number of columns must match data width");
        }
        
        if (index != null && index.size() != rows) {
            throw new IllegalArgumentException(
                "Index size must match data height");
        }
        
        this.index = index != null ? index : Index.range(rows);
        this.columns = new LinkedHashMap<>();
        
        // Extract each column as a Series
        for (int i = 0; i < cols; i++) {
            double[] columnData = new double[rows];
            for (int j = 0; j < rows; j++) {
                columnData[j] = data.get(j, i);
            }
            NDArray columnArray = new NDArray(columnData, rows);
            Series series = new Series(columnArray, this.index, columns[i]);
            this.columns.put(columns[i], series);
        }
    }
    
    /**
     * Gets a column as a Series.
     * 
     * @param columnName the column name
     * @return the Series
     */
    public Series getColumn(String columnName) {
        if (!columns.containsKey(columnName)) {
            throw new IllegalArgumentException("Column not found: " + columnName);
        }
        return columns.get(columnName);
    }
    
    /**
     * Gets multiple columns as a new DataFrame.
     * 
     * @param columnNames the column names
     * @return a new DataFrame
     */
    public DataFrame getColumns(String... columnNames) {
        Map<String, Series> newData = new LinkedHashMap<>();
        for (String col : columnNames) {
            if (!columns.containsKey(col)) {
                throw new IllegalArgumentException("Column not found: " + col);
            }
            newData.put(col, columns.get(col));
        }
        return new DataFrame(newData);
    }
    
    /**
     * Gets the row index.
     * 
     * @return the index
     */
    public Index getIndex() {
        return index;
    }
    
    /**
     * Gets all column names.
     * 
     * @return a list of column names
     */
    public List<String> getColumnNames() {
        return new ArrayList<>(columns.keySet());
    }
    
    /**
     * Gets the number of rows.
     * 
     * @return the number of rows
     */
    public int getRowCount() {
        return index.size();
    }
    
    /**
     * Gets the number of columns.
     * 
     * @return the number of columns
     */
    public int getColumnCount() {
        return columns.size();
    }
    
    /**
     * Gets the shape of the DataFrame (rows, columns).
     * 
     * @return an array with [rows, columns]
     */
    public int[] getShape() {
        return new int[]{getRowCount(), getColumnCount()};
    }
    
    /**
     * Converts the DataFrame to an NDArray.
     * 
     * @return a 2D NDArray
     */
    public NDArray toNDArray() {
        int rows = getRowCount();
        int cols = getColumnCount();
        NDArray result = new NDArray(rows, cols);
        
        int colIdx = 0;
        for (Series series : columns.values()) {
            for (int row = 0; row < rows; row++) {
                result.set(series.get(row), row, colIdx);
            }
            colIdx++;
        }
        
        return result;
    }
    
    /**
     * Gets a value at the specified row and column.
     * 
     * @param rowLabel the row label
     * @param columnName the column name
     * @return the value
     */
    public double at(Object rowLabel, String columnName) {
        Series column = getColumn(columnName);
        return column.get(rowLabel);
    }
    
    /**
     * Gets a value at the specified integer position.
     * 
     * @param row the row position
     * @param column the column position
     * @return the value
     */
    public double iat(int row, int column) {
        String[] colNames = getColumnNames().toArray(new String[0]);
        return getColumn(colNames[column]).get(row);
    }
    
    /**
     * Label-based selection (loc).
     * 
     * @param rowLabels the row labels
     * @param columnNames the column names
     * @return a new DataFrame
     */
    public DataFrame loc(Object[] rowLabels, String... columnNames) {
        Map<String, Series> newData = new LinkedHashMap<>();
        
        for (String colName : columnNames) {
            Series column = getColumn(colName);
            List<Double> values = new ArrayList<>();
            List<Object> newLabels = new ArrayList<>();
            
            for (Object rowLabel : rowLabels) {
                values.add(column.get(rowLabel));
                newLabels.add(rowLabel);
            }
            
            double[] newDataArray = new double[values.size()];
            for (int i = 0; i < values.size(); i++) {
                newDataArray[i] = values.get(i);
            }
            
            Series newSeries = new Series(
                new NDArray(newDataArray, newDataArray.length),
                new Index(newLabels),
                colName
            );
            newData.put(colName, newSeries);
        }
        
        return new DataFrame(newData);
    }
    
    /**
     * Integer position-based selection (iloc).
     * 
     * @param rows the row positions
     * @param columns the column positions
     * @return a new DataFrame
     */
    public DataFrame iloc(int[] rows, int... columns) {
        String[] colNames = getColumnNames().toArray(new String[0]);
        String[] selectedCols = new String[columns.length];
        for (int i = 0; i < columns.length; i++) {
            selectedCols[i] = colNames[columns[i]];
        }
        
        Map<String, Series> newData = new LinkedHashMap<>();
        List<Object> newLabels = new ArrayList<>();
        
        for (String colName : selectedCols) {
            Series column = getColumn(colName);
            double[] newDataArray = new double[rows.length];
            
            for (int i = 0; i < rows.length; i++) {
                newDataArray[i] = column.get(rows[i]);
                if (i == 0 || newLabels.size() < rows.length) {
                    newLabels.add(index.get(rows[i]));
                }
            }
            
            Series newSeries = new Series(
                new NDArray(newDataArray, newDataArray.length),
                new Index(newLabels),
                colName
            );
            newData.put(colName, newSeries);
        }
        
        return new DataFrame(newData);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("DataFrame(");
        sb.append("shape=(").append(getRowCount()).append(", ")
          .append(getColumnCount()).append("))\n");
        
        // Display column names
        sb.append("Columns: ").append(getColumnNames()).append("\n");
        
        // Display first few rows
        int displayRows = Math.min(10, getRowCount());
        for (int i = 0; i < displayRows; i++) {
            sb.append(index.get(i)).append(": ");
            for (String col : getColumnNames()) {
                sb.append(col).append("=").append(getColumn(col).get(i)).append("  ");
            }
            sb.append("\n");
        }
        if (getRowCount() > displayRows) {
            sb.append("...\n");
        }
        
        return sb.toString();
    }
}

