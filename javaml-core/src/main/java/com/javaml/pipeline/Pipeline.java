package com.javaml.pipeline;

import com.javaml.array.NDArray;
import com.javaml.base.Estimator;
import com.javaml.base.Transformer;

import java.util.*;

/**
 * Pipeline for chaining transformers and estimators.
 * Equivalent to scikit-learn's Pipeline.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class Pipeline implements Estimator {
    
    private final List<Step> steps;
    private boolean fitted = false;
    
    /**
     * Creates a new empty Pipeline.
     */
    public Pipeline() {
        this.steps = new ArrayList<>();
    }
    
    /**
     * Adds a step to the pipeline.
     * 
     * @param name the name of the step
     * @param transformer the transformer or estimator
     * @return this pipeline for method chaining
     */
    public Pipeline addStep(String name, Object transformer) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Step name cannot be null or empty");
        }
        if (transformer == null) {
            throw new IllegalArgumentException("Transformer cannot be null");
        }
        
        steps.add(new Step(name, transformer));
        return this;
    }
    
    @Override
    public void fit(NDArray X, NDArray y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (steps.isEmpty()) {
            throw new IllegalStateException("Pipeline must have at least one step");
        }
        
        NDArray currentX = X;
        
        // Fit all transformers
        for (int i = 0; i < steps.size() - 1; i++) {
            Step step = steps.get(i);
            if (step.transformer instanceof Transformer) {
                Transformer transformer = (Transformer) step.transformer;
                transformer.fit(currentX);
                currentX = transformer.transform(currentX);
            } else {
                throw new IllegalArgumentException(
                    "All steps except the last must be transformers");
            }
        }
        
        // Fit the final estimator
        Step finalStep = steps.get(steps.size() - 1);
        if (finalStep.transformer instanceof Estimator) {
            Estimator estimator = (Estimator) finalStep.transformer;
            estimator.fit(currentX, y);
        } else if (finalStep.transformer instanceof Transformer) {
            Transformer transformer = (Transformer) finalStep.transformer;
            transformer.fit(currentX);
        } else {
            throw new IllegalArgumentException(
                "Final step must be an estimator or transformer");
        }
        
        fitted = true;
    }
    
    @Override
    public NDArray predict(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before prediction");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        
        NDArray currentX = X;
        
        // Transform through all transformers
        for (int i = 0; i < steps.size() - 1; i++) {
            Step step = steps.get(i);
            if (step.transformer instanceof Transformer) {
                Transformer transformer = (Transformer) step.transformer;
                currentX = transformer.transform(currentX);
            }
        }
        
        // Predict with final estimator
        Step finalStep = steps.get(steps.size() - 1);
        if (finalStep.transformer instanceof Estimator) {
            Estimator estimator = (Estimator) finalStep.transformer;
            return estimator.predict(currentX);
        } else {
            throw new IllegalStateException("Final step must be an estimator for prediction");
        }
    }
    
    @Override
    public double score(NDArray X, NDArray y) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before scoring");
        }
        
        NDArray predictions = predict(X);
        
        Step finalStep = steps.get(steps.size() - 1);
        if (finalStep.transformer instanceof Estimator) {
            Estimator estimator = (Estimator) finalStep.transformer;
            return estimator.score(X, y);
        } else {
            // Calculate accuracy for classification
            return calculateAccuracy(predictions, y);
        }
    }
    
    /**
     * Calculates accuracy.
     */
    private double calculateAccuracy(NDArray predictions, NDArray y) {
        int correct = 0;
        for (int i = 0; i < predictions.getSize(); i++) {
            if (Math.abs(predictions.get(i) - y.get(i)) < 1e-6) {
                correct++;
            }
        }
        return (double) correct / predictions.getSize();
    }
    
    /**
     * Gets the step at the specified index.
     * 
     * @param index the index
     * @return the step name
     */
    public String getStepName(int index) {
        if (index < 0 || index >= steps.size()) {
            throw new IndexOutOfBoundsException("Step index out of bounds");
        }
        return steps.get(index).name;
    }
    
    /**
     * Gets the transformer/estimator at the specified index.
     * 
     * @param index the index
     * @return the transformer or estimator
     */
    public Object getStep(int index) {
        if (index < 0 || index >= steps.size()) {
            throw new IndexOutOfBoundsException("Step index out of bounds");
        }
        return steps.get(index).transformer;
    }
    
    /**
     * Gets the number of steps.
     * 
     * @return the number of steps
     */
    public int getNSteps() {
        return steps.size();
    }
    
    /**
     * Internal step class.
     */
    private static class Step {
        final String name;
        final Object transformer;
        
        Step(String name, Object transformer) {
            this.name = name;
            this.transformer = transformer;
        }
    }
}

