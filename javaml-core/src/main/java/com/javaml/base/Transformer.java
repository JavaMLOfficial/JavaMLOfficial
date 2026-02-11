package com.javaml.base;

import com.javaml.array.NDArray;

/**
 * Interface for transformers that can transform data.
 * Equivalent to scikit-learn's TransformerMixin.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public interface Transformer {
    
    /**
     * Fits the transformer to the data.
     * 
     * @param X the input data
     */
    void fit(NDArray X);
    
    /**
     * Transforms the data.
     * 
     * @param X the input data
     * @return the transformed data
     */
    NDArray transform(NDArray X);
    
    /**
     * Fits the transformer and transforms the data in one step.
     * 
     * @param X the input data
     * @return the transformed data
     */
    default NDArray fitTransform(NDArray X) {
        fit(X);
        return transform(X);
    }
}

