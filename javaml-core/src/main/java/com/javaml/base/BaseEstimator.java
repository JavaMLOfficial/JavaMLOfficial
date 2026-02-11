package com.javaml.base;

import java.util.Map;

/**
 * Base class for all estimators in JavaML.
 * Equivalent to scikit-learn's BaseEstimator.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public abstract class BaseEstimator {
    
    /**
     * Gets the parameters of this estimator.
     * 
     * @return a map of parameter names to values
     */
    public abstract Map<String, Object> getParams();
    
    /**
     * Sets the parameters of this estimator.
     * 
     * @param params a map of parameter names to values
     */
    public abstract void setParams(Map<String, Object> params);
}

