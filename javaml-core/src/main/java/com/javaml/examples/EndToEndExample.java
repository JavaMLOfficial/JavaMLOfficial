package com.javaml.examples;

import com.javaml.array.*;
import com.javaml.dataframe.DataFrame;
import com.javaml.dataframe.Index;
import com.javaml.io.CSVReader;
import com.javaml.linear.LinearRegression;
import com.javaml.model_selection.ModelSelection;
import com.javaml.preprocessing.StandardScaler;

import java.io.IOException;

/**
 * End-to-end example demonstrating JavaML usage.
 * Shows a complete machine learning workflow from data loading to model evaluation.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class EndToEndExample {
    
    public static void main(String[] args) {
        System.out.println("=== JavaML End-to-End Example ===\n");
        
        // 1. Create sample data
        System.out.println("1. Creating sample data...");
        NDArray X = ArrayCreation.randn(100, 3); // 100 samples, 3 features
        NDArray y = createTarget(X); // Create target variable
        
        System.out.println("   X shape: " + java.util.Arrays.toString(X.getShape()));
        System.out.println("   y shape: " + java.util.Arrays.toString(y.getShape()));
        
        // 2. Create DataFrame
        System.out.println("\n2. Creating DataFrame...");
        DataFrame df = new DataFrame(X, Index.range(100), "feature1", "feature2", "feature3");
        System.out.println("   DataFrame shape: " + java.util.Arrays.toString(df.getShape()));
        
        // 3. Data preprocessing
        System.out.println("\n3. Preprocessing data...");
        StandardScaler scaler = new StandardScaler();
        scaler.fit(X);
        NDArray X_scaled = scaler.transform(X);
        System.out.println("   Data scaled. Mean: " + ArrayStats.mean(X_scaled.flatten()));
        
        // 4. Train-test split
        System.out.println("\n4. Splitting data...");
        NDArray[] split = ModelSelection.trainTestSplit(X_scaled, y, 0.2, 42L);
        NDArray X_train = split[0];
        NDArray X_test = split[1];
        NDArray y_train = split[2];
        NDArray y_test = split[3];
        
        System.out.println("   Train size: " + X_train.getShape()[0]);
        System.out.println("   Test size: " + X_test.getShape()[0]);
        
        // 5. Train model
        System.out.println("\n5. Training Linear Regression model...");
        LinearRegression model = new LinearRegression();
        model.fit(X_train, y_train);
        System.out.println("   Model trained successfully");
        
        // 6. Make predictions
        System.out.println("\n6. Making predictions...");
        NDArray y_pred = model.predict(X_test);
        System.out.println("   Predictions made for " + y_pred.getSize() + " samples");
        
        // 7. Evaluate model
        System.out.println("\n7. Evaluating model...");
        double score = model.score(X_test, y_test);
        System.out.println("   R² Score: " + String.format("%.4f", score));
        
        // 8. Statistical analysis
        System.out.println("\n8. Statistical analysis...");
        System.out.println("   Test target mean: " + String.format("%.4f", ArrayStats.mean(y_test)));
        System.out.println("   Predictions mean: " + String.format("%.4f", ArrayStats.mean(y_pred)));
        System.out.println("   Test target std: " + String.format("%.4f", ArrayStats.std(y_test)));
        System.out.println("   Predictions std: " + String.format("%.4f", ArrayStats.std(y_pred)));
        
        // 9. Array operations demonstration
        System.out.println("\n9. Demonstrating array operations...");
        NDArray arr1 = ArrayCreation.arange(0, 10);
        NDArray arr2 = ArrayCreation.ones(10);
        NDArray arr3 = ArrayMath.add(arr1, arr2);
        System.out.println("   arange(0, 10) + ones(10) = " + 
            java.util.Arrays.toString(arr3.getData()));
        
        // 10. Linear algebra demonstration
        System.out.println("\n10. Demonstrating linear algebra...");
        NDArray matrix = ArrayCreation.randn(3, 3);
        double det = LinearAlgebra.det(matrix);
        System.out.println("   Determinant of 3x3 random matrix: " + String.format("%.4f", det));
        
        System.out.println("\n=== Example Complete ===");
    }
    
    /**
     * Creates a target variable from features (simple linear relationship).
     */
    private static NDArray createTarget(NDArray X) {
        int[] shape = X.getShape();
        int nSamples = shape[0];
        double[] target = new double[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            // Simple linear relationship: y = 2*x1 + 3*x2 - x3 + noise
            double value = 2.0 * X.get(i, 0) + 3.0 * X.get(i, 1) - X.get(i, 2);
            // Add some noise
            value += ArrayCreation.randn(1).get(0) * 0.1;
            target[i] = value;
        }
        
        return new NDArray(target, target.length);
    }
}

