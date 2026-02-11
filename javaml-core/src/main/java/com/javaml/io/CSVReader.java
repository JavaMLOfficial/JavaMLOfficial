package com.javaml.io;

import com.javaml.dataframe.DataFrame;
import com.javaml.dataframe.Index;
import com.javaml.dataframe.Series;
import com.javaml.array.NDArray;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * CSV file reading and writing utilities for DataFrame.
 * Equivalent to Pandas read_csv() and to_csv().
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class CSVReader {
    
    private CSVReader() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Reads a CSV file into a DataFrame.
     * 
     * @param filePath the path to the CSV file
     * @return a DataFrame
     * @throws IOException if an I/O error occurs
     */
    public static DataFrame readCSV(String filePath) throws IOException {
        return readCSV(Path.of(filePath));
    }
    
    /**
     * Reads a CSV file into a DataFrame.
     * 
     * @param filePath the path to the CSV file
     * @return a DataFrame
     * @throws IOException if an I/O error occurs
     */
    public static DataFrame readCSV(Path filePath) throws IOException {
        return readCSV(filePath, ',', true, 0);
    }
    
    /**
     * Reads a CSV file into a DataFrame with custom options.
     * 
     * @param filePath the path to the CSV file
     * @param delimiter the delimiter character
     * @param hasHeader whether the file has a header row
     * @param skipRows number of rows to skip
     * @return a DataFrame
     * @throws IOException if an I/O error occurs
     */
    public static DataFrame readCSV(Path filePath, char delimiter, boolean hasHeader, int skipRows) 
            throws IOException {
        List<String> lines = Files.readAllLines(filePath);
        
        if (lines.isEmpty()) {
            throw new IllegalArgumentException("CSV file is empty");
        }
        
        // Skip rows if needed
        int startRow = skipRows;
        if (hasHeader && startRow == 0) {
            startRow = 1;
        }
        
        // Parse header
        String[] columnNames;
        if (hasHeader && lines.size() > 0) {
            columnNames = parseLine(lines.get(0), delimiter);
        } else {
            // Generate column names
            String[] firstRow = parseLine(lines.get(startRow), delimiter);
            columnNames = new String[firstRow.length];
            for (int i = 0; i < firstRow.length; i++) {
                columnNames[i] = "col" + i;
            }
        }
        
        // Parse data rows
        List<List<Double>> columns = new ArrayList<>();
        for (int i = 0; i < columnNames.length; i++) {
            columns.add(new ArrayList<>());
        }
        
        List<Object> rowLabels = new ArrayList<>();
        
        for (int i = startRow; i < lines.size(); i++) {
            String[] values = parseLine(lines.get(i), delimiter);
            
            if (values.length != columnNames.length) {
                // Skip rows with inconsistent column count
                continue;
            }
            
            // Use row number as label
            rowLabels.add(i - startRow);
            
            for (int j = 0; j < columnNames.length; j++) {
                try {
                    double value = Double.parseDouble(values[j].trim());
                    columns.get(j).add(value);
                } catch (NumberFormatException e) {
                    // Handle non-numeric values (for now, use 0.0 or NaN)
                    columns.get(j).add(Double.NaN);
                }
            }
        }
        
        // Create Series for each column
        Map<String, Series> data = new LinkedHashMap<>();
        Index index = new Index(rowLabels);
        
        for (int i = 0; i < columnNames.length; i++) {
            double[] columnData = new double[columns.get(i).size()];
            for (int j = 0; j < columns.get(i).size(); j++) {
                columnData[j] = columns.get(i).get(j);
            }
            NDArray columnArray = new NDArray(columnData, columnData.length);
            Series series = new Series(columnArray, index, columnNames[i]);
            data.put(columnNames[i], series);
        }
        
        return new DataFrame(data);
    }
    
    /**
     * Writes a DataFrame to a CSV file.
     * 
     * @param df the DataFrame to write
     * @param filePath the path to the output file
     * @throws IOException if an I/O error occurs
     */
    public static void toCSV(DataFrame df, String filePath) throws IOException {
        toCSV(df, Path.of(filePath));
    }
    
    /**
     * Writes a DataFrame to a CSV file.
     * 
     * @param df the DataFrame to write
     * @param filePath the path to the output file
     * @throws IOException if an I/O error occurs
     */
    public static void toCSV(DataFrame df, Path filePath) throws IOException {
        toCSV(df, filePath, ',', true);
    }
    
    /**
     * Writes a DataFrame to a CSV file with custom options.
     * 
     * @param df the DataFrame to write
     * @param filePath the path to the output file
     * @param delimiter the delimiter character
     * @param includeHeader whether to include header row
     * @throws IOException if an I/O error occurs
     */
    public static void toCSV(DataFrame df, Path filePath, char delimiter, boolean includeHeader) 
            throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(filePath)) {
            // Write header
            if (includeHeader) {
                List<String> columnNames = df.getColumnNames();
                writer.write(String.join(String.valueOf(delimiter), columnNames));
                writer.newLine();
            }
            
            // Write data rows
            int rowCount = df.getRowCount();
            List<String> columnNames = df.getColumnNames();
            
            for (int i = 0; i < rowCount; i++) {
                List<String> values = new ArrayList<>();
                for (String colName : columnNames) {
                    double value = df.getColumn(colName).get(i);
                    values.add(String.valueOf(value));
                }
                writer.write(String.join(String.valueOf(delimiter), values));
                writer.newLine();
            }
        }
    }
    
    /**
     * Parses a CSV line into an array of strings.
     */
    private static String[] parseLine(String line, char delimiter) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        
        for (char c : line.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == delimiter && !inQuotes) {
                fields.add(current.toString().trim());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString().trim());
        
        return fields.toArray(new String[0]);
    }
}

