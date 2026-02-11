package com.javaml.examples;

import com.javaml.array.*;
import com.javaml.linear.LinearRegression;
import com.javaml.linear.LogisticRegression;
import com.javaml.metrics.Metrics;
import com.javaml.model_selection.ModelSelection;
import com.javaml.pipeline.Pipeline;
import com.javaml.preprocessing.MinMaxScaler;
import com.javaml.preprocessing.OneHotEncoder;
import com.javaml.preprocessing.StandardScaler;
import com.javaml.tree.DecisionTreeClassifier;
import com.javaml.tree.RandomForestClassifier;

/**
 * Complete machine learning workflow example demonstrating:
 * - Data preprocessing
 * - Multiple ML algorithms
 * - Pipeline usage
 * - Model evaluation
 * - Metrics calculation
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class CompleteMLWorkflow {
    
    public static void main(String[] args) {
        System.out.println("=== Complete ML Workflow Example ===\n");
        
        // ========== REGRESSION EXAMPLE ==========
        System.out.println("1. REGRESSION WORKFLOW");
        System.out.println("=".repeat(50));
        regressionExample();
        
        System.out.println("\n");
        
        // ========== CLASSIFICATION EXAMPLE ==========
        System.out.println("2. CLASSIFICATION WORKFLOW");
        System.out.println("=".repeat(50));
        classificationExample();
        
        System.out.println("\n");
        
        // ========== PIPELINE EXAMPLE ==========
        System.out.println("3. PIPELINE WORKFLOW");
        System.out.println("=".repeat(50));
        pipelineExample();
        
        System.out.println("\n=== All Examples Complete ===");
    }
    
    /**
     * Regression workflow example.
     */
    private static void regressionExample() {
        // Generate synthetic regression data
        System.out.println("Generating regression dataset...");
        NDArray X = ArrayCreation.randn(200, 3);
        NDArray y = createRegressionTarget(X);
        
        System.out.println("  Dataset: " + X.getShape()[0] + " samples, " + 
                          X.getShape()[1] + " features");
        
        // Preprocess
        System.out.println("Preprocessing data...");
        StandardScaler scaler = new StandardScaler();
        scaler.fit(X);
        NDArray X_scaled = scaler.transform(X);
        
        // Split data
        System.out.println("Splitting data...");
        NDArray[] split = ModelSelection.trainTestSplit(X_scaled, y, 0.2, 42L);
        NDArray X_train = split[0];
        NDArray X_test = split[1];
        NDArray y_train = split[2];
        NDArray y_test = split[3];
        
        System.out.println("  Train: " + X_train.getShape()[0] + " samples");
        System.out.println("  Test: " + X_test.getShape()[0] + " samples");
        
        // Train model
        System.out.println("Training Linear Regression...");
        LinearRegression model = new LinearRegression();
        model.fit(X_train, y_train);
        
        // Evaluate
        System.out.println("Evaluating model...");
        NDArray y_pred = model.predict(X_test);
        double r2 = Metrics.r2Score(y_test, y_pred);
        double mse = Metrics.meanSquaredError(y_test, y_pred);
        double mae = Metrics.meanAbsoluteError(y_test, y_pred);
        
        System.out.println("  R² Score: " + String.format("%.4f", r2));
        System.out.println("  MSE: " + String.format("%.4f", mse));
        System.out.println("  MAE: " + String.format("%.4f", mae));
    }
    
    /**
     * Classification workflow example.
     */
    private static void classificationExample() {
        // Generate synthetic classification data
        System.out.println("Generating classification dataset...");
        NDArray X = ArrayCreation.randn(300, 4);
        NDArray y = createClassificationTarget(X);
        
        System.out.println("  Dataset: " + X.getShape()[0] + " samples, " + 
                          X.getShape()[1] + " features");
        
        // Preprocess
        System.out.println("Preprocessing data...");
        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(X);
        NDArray X_scaled = scaler.transform(X);
        
        // Split data
        System.out.println("Splitting data...");
        NDArray[] split = ModelSelection.trainTestSplit(X_scaled, y, 0.3, 42L);
        NDArray X_train = split[0];
        NDArray X_test = split[1];
        NDArray y_train = split[2];
        NDArray y_test = split[3];
        
        System.out.println("  Train: " + X_train.getShape()[0] + " samples");
        System.out.println("  Test: " + X_test.getShape()[0] + " samples");
        
        // Train multiple models
        System.out.println("Training models...");
        
        // Logistic Regression
        System.out.println("  Training Logistic Regression...");
        LogisticRegression lr = new LogisticRegression(0.01, 1000);
        lr.fit(X_train, y_train);
        NDArray lr_pred = lr.predict(X_test);
        double lr_acc = Metrics.accuracyScore(y_test, lr_pred);
        System.out.println("    Accuracy: " + String.format("%.4f", lr_acc));
        
        // Decision Tree
        System.out.println("  Training Decision Tree...");
        DecisionTreeClassifier dt = new DecisionTreeClassifier(5, 10, 5);
        dt.fit(X_train, y_train);
        NDArray dt_pred = dt.predict(X_test);
        double dt_acc = Metrics.accuracyScore(y_test, dt_pred);
        System.out.println("    Accuracy: " + String.format("%.4f", dt_acc));
        
        // Random Forest
        System.out.println("  Training Random Forest...");
        RandomForestClassifier rf = new RandomForestClassifier(50, -1);
        rf.fit(X_train, y_train);
        NDArray rf_pred = rf.predict(X_test);
        double rf_acc = Metrics.accuracyScore(y_test, rf_pred);
        System.out.println("    Accuracy: " + String.format("%.4f", rf_acc));
        
        // Confusion matrix for best model
        System.out.println("Confusion Matrix (Random Forest):");
        NDArray cm = Metrics.confusionMatrix(y_test, rf_pred);
        System.out.println("  " + cm.toString());
    }
    
    /**
     * Pipeline workflow example.
     */
    private static void pipelineExample() {
        System.out.println("Creating ML Pipeline...");
        
        // Generate data
        NDArray X = ArrayCreation.randn(200, 3);
        NDArray y = createClassificationTarget(X);
        
        // Split
        NDArray[] split = ModelSelection.trainTestSplit(X, y, 0.2, 42L);
        NDArray X_train = split[0];
        NDArray X_test = split[1];
        NDArray y_train = split[2];
        NDArray y_test = split[3];
        
        // Create pipeline: StandardScaler -> LogisticRegression
        System.out.println("Building pipeline: StandardScaler -> LogisticRegression");
        Pipeline pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression());
        
        // Fit pipeline
        System.out.println("Fitting pipeline...");
        pipeline.fit(X_train, y_train);
        
        // Evaluate
        System.out.println("Evaluating pipeline...");
        double score = pipeline.score(X_test, y_test);
        NDArray predictions = pipeline.predict(X_test);
        double accuracy = Metrics.accuracyScore(y_test, predictions);
        
        System.out.println("  Pipeline Score: " + String.format("%.4f", score));
        System.out.println("  Accuracy: " + String.format("%.4f", accuracy));
    }
    
    /**
     * Creates regression target.
     */
    private static NDArray createRegressionTarget(NDArray X) {
        int[] shape = X.getShape();
        double[] target = new double[shape[0]];
        
        for (int i = 0; i < shape[0]; i++) {
            target[i] = 2.0 * X.get(i, 0) + 3.0 * X.get(i, 1) - X.get(i, 2) + 
                       ArrayCreation.randn(1).get(0) * 0.5;
        }
        
        return new NDArray(target, target.length);
    }
    
    /**
     * Creates classification target.
     */
    private static NDArray createClassificationTarget(NDArray X) {
        int[] shape = X.getShape();
        double[] target = new double[shape[0]];
        
        for (int i = 0; i < shape[0]; i++) {
            double sum = 0.0;
            for (int j = 0; j < shape[1]; j++) {
                sum += X.get(i, j);
            }
            target[i] = sum > 0 ? 1.0 : 0.0;
        }
        
        return new NDArray(target, target.length);
    }
}

