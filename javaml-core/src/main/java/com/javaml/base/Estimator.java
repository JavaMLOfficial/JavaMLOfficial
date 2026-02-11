package com.javaml.base;

import com.javaml.array.NDArray;

/**
 * Interface for estimators that can make predictions.
 * Equivalent to scikit-learn's Estimator interface.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public interface Estimator {
    
    /**
     * Fits the estimator to the training data.
     * 
     * @param X the training features
     * @param y the training targets
     */
    void fit(NDArray X, NDArray y);
    
    /**
     * Makes predictions on the input data.
     * 
     * @param X the input data
     * @return the predictions
     */
    NDArray predict(NDArray X);
    
    /**
     * Returns the score of the model on the given data.
     * 
     * @param X the input data
     * @param y the true targets
     * @return the score
     */
    double score(NDArray X, NDArray y);
}

