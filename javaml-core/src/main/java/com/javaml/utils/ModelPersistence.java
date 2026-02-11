package com.javaml.utils;

import com.javaml.base.BaseEstimator;

import java.io.*;
import java.util.Base64;

/**
 * Model persistence utilities for saving and loading ML models.
 * Equivalent to scikit-learn's joblib.dump() and joblib.load().
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public final class ModelPersistence {
    
    private ModelPersistence() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Saves a model to a file.
     * 
     * @param model the model to save
     * @param filePath the file path to save to
     * @throws IOException if an I/O error occurs
     */
    public static void dump(BaseEstimator model, String filePath) throws IOException {
        if (model == null) {
            throw new IllegalArgumentException("Model cannot be null");
        }
        if (filePath == null || filePath.isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(filePath))) {
            oos.writeObject(model);
        }
    }
    
    /**
     * Loads a model from a file.
     * 
     * @param filePath the file path to load from
     * @return the loaded model
     * @throws IOException if an I/O error occurs
     * @throws ClassNotFoundException if the class cannot be found
     */
    public static BaseEstimator load(String filePath) 
            throws IOException, ClassNotFoundException {
        if (filePath == null || filePath.isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(filePath))) {
            return (BaseEstimator) ois.readObject();
        }
    }
    
    /**
     * Serializes a model to a Base64 string.
     * 
     * @param model the model to serialize
     * @return Base64 encoded string
     * @throws IOException if an I/O error occurs
     */
    public static String dumpToString(BaseEstimator model) throws IOException {
        if (model == null) {
            throw new IllegalArgumentException("Model cannot be null");
        }
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(model);
        }
        
        return Base64.getEncoder().encodeToString(baos.toByteArray());
    }
    
    /**
     * Deserializes a model from a Base64 string.
     * 
     * @param encodedModel the Base64 encoded model string
     * @return the deserialized model
     * @throws IOException if an I/O error occurs
     * @throws ClassNotFoundException if the class cannot be found
     */
    public static BaseEstimator loadFromString(String encodedModel) 
            throws IOException, ClassNotFoundException {
        if (encodedModel == null || encodedModel.isEmpty()) {
            throw new IllegalArgumentException("Encoded model cannot be null or empty");
        }
        
        byte[] data = Base64.getDecoder().decode(encodedModel);
        ByteArrayInputStream bais = new ByteArrayInputStream(data);
        
        try (ObjectInputStream ois = new ObjectInputStream(bais)) {
            return (BaseEstimator) ois.readObject();
        }
    }
}

