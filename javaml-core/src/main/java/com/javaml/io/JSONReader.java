package com.javaml.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.javaml.dataframe.DataFrame;
import com.javaml.dataframe.Index;
import com.javaml.dataframe.Series;
import com.javaml.array.NDArray;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * JSON file reading and writing utilities for DataFrame.
 * Equivalent to Pandas read_json() and to_json().
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class JSONReader {
    
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    private JSONReader() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Reads a JSON file into a DataFrame.
     * Supports JSON arrays of objects format.
     * 
     * @param filePath the path to the JSON file
     * @return a DataFrame
     * @throws IOException if an I/O error occurs
     */
    public static DataFrame readJSON(String filePath) throws IOException {
        return readJSON(Path.of(filePath));
    }
    
    /**
     * Reads a JSON file into a DataFrame.
     * 
     * @param filePath the path to the JSON file
     * @return a DataFrame
     * @throws IOException if an I/O error occurs
     */
    public static DataFrame readJSON(Path filePath) throws IOException {
        String content = Files.readString(filePath);
        JsonNode root = objectMapper.readTree(content);
        
        if (root.isArray()) {
            return readJSONArray(root);
        } else if (root.isObject()) {
            return readJSONObject(root);
        } else {
            throw new IllegalArgumentException("JSON must be an array or object");
        }
    }
    
    /**
     * Reads a JSON array into a DataFrame.
     */
    private static DataFrame readJSONArray(JsonNode arrayNode) {
        if (!arrayNode.isArray() || arrayNode.size() == 0) {
            throw new IllegalArgumentException("JSON array cannot be empty");
        }
        
        // Get all field names from first object
        JsonNode firstObject = arrayNode.get(0);
        if (!firstObject.isObject()) {
            throw new IllegalArgumentException("JSON array must contain objects");
        }
        
        Set<String> fieldNames = new LinkedHashSet<>();
        firstObject.fieldNames().forEachRemaining(fieldNames::add);
        
        // Collect data for each field
        Map<String, List<Double>> columns = new LinkedHashMap<>();
        for (String fieldName : fieldNames) {
            columns.put(fieldName, new ArrayList<>());
        }
        
        List<Object> rowLabels = new ArrayList<>();
        
        for (int i = 0; i < arrayNode.size(); i++) {
            JsonNode obj = arrayNode.get(i);
            rowLabels.add(i);
            
            for (String fieldName : fieldNames) {
                JsonNode value = obj.get(fieldName);
                double numValue = value != null && value.isNumber() 
                    ? value.asDouble() 
                    : Double.NaN;
                columns.get(fieldName).add(numValue);
            }
        }
        
        // Create DataFrame
        Map<String, Series> data = new LinkedHashMap<>();
        Index index = new Index(rowLabels);
        
        for (Map.Entry<String, List<Double>> entry : columns.entrySet()) {
            double[] columnData = new double[entry.getValue().size()];
            for (int i = 0; i < entry.getValue().size(); i++) {
                columnData[i] = entry.getValue().get(i);
            }
            NDArray columnArray = new NDArray(columnData, columnData.length);
            Series series = new Series(columnArray, index, entry.getKey());
            data.put(entry.getKey(), series);
        }
        
        return new DataFrame(data);
    }
    
    /**
     * Reads a JSON object into a DataFrame (object of arrays format).
     */
    private static DataFrame readJSONObject(JsonNode objectNode) {
        if (!objectNode.isObject()) {
            throw new IllegalArgumentException("JSON must be an object");
        }
        
        Map<String, List<Double>> columns = new LinkedHashMap<>();
        int maxSize = 0;
        
        // Extract arrays from object
        objectNode.fields().forEachRemaining(entry -> {
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            
            if (value.isArray()) {
                List<Double> values = new ArrayList<>();
                for (JsonNode node : value) {
                    values.add(node.isNumber() ? node.asDouble() : Double.NaN);
                }
                columns.put(key, values);
            }
        });
        
        // Find maximum size
        for (List<Double> list : columns.values()) {
            maxSize = Math.max(maxSize, list.size());
        }
        
        // Pad columns to same size
        for (List<Double> list : columns.values()) {
            while (list.size() < maxSize) {
                list.add(Double.NaN);
            }
        }
        
        // Create DataFrame
        Map<String, Series> data = new LinkedHashMap<>();
        List<Object> rowLabels = new ArrayList<>();
        for (int i = 0; i < maxSize; i++) {
            rowLabels.add(i);
        }
        Index index = new Index(rowLabels);
        
        for (Map.Entry<String, List<Double>> entry : columns.entrySet()) {
            double[] columnData = new double[entry.getValue().size()];
            for (int i = 0; i < entry.getValue().size(); i++) {
                columnData[i] = entry.getValue().get(i);
            }
            NDArray columnArray = new NDArray(columnData, columnData.length);
            Series series = new Series(columnArray, index, entry.getKey());
            data.put(entry.getKey(), series);
        }
        
        return new DataFrame(data);
    }
    
    /**
     * Writes a DataFrame to a JSON file.
     * 
     * @param df the DataFrame to write
     * @param filePath the path to the output file
     * @throws IOException if an I/O error occurs
     */
    public static void toJSON(DataFrame df, String filePath) throws IOException {
        toJSON(df, Path.of(filePath));
    }
    
    /**
     * Writes a DataFrame to a JSON file.
     * 
     * @param df the DataFrame to write
     * @param filePath the path to the output file
     * @throws IOException if an I/O error occurs
     */
    public static void toJSON(DataFrame df, Path filePath) throws IOException {
        Map<String, Object> jsonObject = new LinkedHashMap<>();
        
        List<String> columnNames = df.getColumnNames();
        int rowCount = df.getRowCount();
        
        // Create object of arrays format
        for (String colName : columnNames) {
            List<Double> values = new ArrayList<>();
            Series series = df.getColumn(colName);
            for (int i = 0; i < rowCount; i++) {
                values.add(series.get(i));
            }
            jsonObject.put(colName, values);
        }
        
        String json = objectMapper.writerWithDefaultPrettyPrinter()
            .writeValueAsString(jsonObject);
        
        Files.writeString(filePath, json);
    }
}

