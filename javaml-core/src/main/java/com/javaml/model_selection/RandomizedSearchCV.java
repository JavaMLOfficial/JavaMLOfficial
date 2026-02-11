package com.javaml.model_selection;

import com.javaml.array.NDArray;
import com.javaml.base.Estimator;
import com.javaml.random.RandomGenerator;
import com.javaml.util.VirtualThreadExecutor;

import java.util.*;

/**
 * Randomized search over specified parameter distributions for an estimator.
 * Equivalent to scikit-learn's RandomizedSearchCV.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class RandomizedSearchCV {
    
    private final Estimator estimator;
    private final Map<String, List<Object>> paramDistributions;
    private final int nIter;
    private final int cv;
    private final String scoring;
    private final RandomGenerator random;
    private final VirtualThreadExecutor executor;
    
    private Estimator bestEstimator;
    private Map<String, Object> bestParams;
    private double bestScore;
    private Map<Map<String, Object>, Double> cvResults;
    
    /**
     * Creates a new RandomizedSearchCV.
     * 
     * @param estimator the estimator to optimize
     * @param paramDistributions the parameter distributions to sample from
     * @param nIter number of parameter settings sampled
     * @param cv the number of cross-validation folds
     */
    public RandomizedSearchCV(Estimator estimator, 
                              Map<String, List<Object>> paramDistributions,
                              int nIter, int cv) {
        this(estimator, paramDistributions, nIter, cv, null, null, null);
    }
    
    /**
     * Creates a new RandomizedSearchCV with all parameters.
     * 
     * @param estimator the estimator to optimize
     * @param paramDistributions the parameter distributions to sample from
     * @param nIter number of parameter settings sampled
     * @param cv the number of cross-validation folds
     * @param scoring the scoring metric
     * @param randomState random seed
     * @param executor the virtual thread executor for parallel execution
     */
    public RandomizedSearchCV(Estimator estimator, 
                              Map<String, List<Object>> paramDistributions,
                              int nIter, int cv, String scoring, Long randomState,
                              VirtualThreadExecutor executor) {
        if (estimator == null) {
            throw new IllegalArgumentException("Estimator cannot be null");
        }
        if (paramDistributions == null || paramDistributions.isEmpty()) {
            throw new IllegalArgumentException("Parameter distributions cannot be null or empty");
        }
        if (nIter <= 0) {
            throw new IllegalArgumentException("nIter must be positive");
        }
        if (cv < 2) {
            throw new IllegalArgumentException("cv must be at least 2");
        }
        
        this.estimator = estimator;
        this.paramDistributions = paramDistributions;
        this.nIter = nIter;
        this.cv = cv;
        this.scoring = scoring;
        this.random = randomState != null ? new RandomGenerator(randomState) : RandomGenerator.getDefault();
        this.executor = executor != null ? executor : VirtualThreadExecutor.getDefault();
    }
    
    /**
     * Fits the RandomizedSearchCV to the data.
     * 
     * @param X the training features
     * @param y the training targets
     */
    public void fit(NDArray X, NDArray y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        
        cvResults = new LinkedHashMap<>();
        bestScore = Double.NEGATIVE_INFINITY;
        bestParams = null;
        bestEstimator = null;
        
        // Sample nIter parameter combinations randomly
        for (int iter = 0; iter < nIter; iter++) {
            Map<String, Object> params = sampleParameters(paramDistributions);
            
            try {
                // Create estimator with these parameters
                Estimator candidate = createEstimatorWithParams(estimator, params);
                
                // Perform cross-validation
                double[] scores = ModelSelection.crossValScore(candidate, X, y, cv);
                double meanScore = Arrays.stream(scores).average().orElse(0.0);
                
                cvResults.put(params, meanScore);
                
                // Update best if better
                if (meanScore > bestScore) {
                    bestScore = meanScore;
                    bestParams = params;
                    bestEstimator = candidate;
                }
            } catch (Exception e) {
                // Skip invalid parameter combinations
                cvResults.put(params, Double.NaN);
            }
        }
        
        // Fit best estimator on full dataset
        if (bestEstimator != null) {
            bestEstimator.fit(X, y);
        }
    }
    
    /**
     * Samples a random parameter combination from distributions.
     */
    private Map<String, Object> sampleParameters(Map<String, List<Object>> distributions) {
        Map<String, Object> params = new HashMap<>();
        
        for (Map.Entry<String, List<Object>> entry : distributions.entrySet()) {
            String paramName = entry.getKey();
            List<Object> values = entry.getValue();
            
            if (values == null || values.isEmpty()) {
                continue;
            }
            
            // Randomly select a value
            int idx = random.randint(0, values.size()).get(0);
            params.put(paramName, values.get(idx));
        }
        
        return params;
    }
    
    /**
     * Makes predictions using the best estimator.
     * 
     * @param X the input features
     * @return the predictions
     */
    public NDArray predict(NDArray X) {
        if (bestEstimator == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted before prediction");
        }
        return bestEstimator.predict(X);
    }
    
    /**
     * Returns the score of the best estimator.
     * 
     * @param X the input features
     * @param y the true targets
     * @return the score
     */
    public double score(NDArray X, NDArray y) {
        if (bestEstimator == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted before scoring");
        }
        return bestEstimator.score(X, y);
    }
    
    /**
     * Gets the best estimator.
     * 
     * @return the best estimator
     */
    public Estimator getBestEstimator() {
        if (bestEstimator == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted first");
        }
        return bestEstimator;
    }
    
    /**
     * Gets the best parameters.
     * 
     * @return the best parameters
     */
    public Map<String, Object> getBestParams() {
        if (bestParams == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted first");
        }
        return new HashMap<>(bestParams);
    }
    
    /**
     * Gets the best score.
     * 
     * @return the best score
     */
    public double getBestScore() {
        if (bestParams == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted first");
        }
        return bestScore;
    }
    
    /**
     * Gets the cross-validation results.
     * 
     * @return map of parameter combinations to scores
     */
    public Map<Map<String, Object>, Double> getCvResults() {
        if (cvResults == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted first");
        }
        return new HashMap<>(cvResults);
    }
    
    /**
     * Creates a new estimator instance with the specified parameters.
     */
    private Estimator createEstimatorWithParams(Estimator template, Map<String, Object> params) {
        try {
            // Try to clone or create new instance
            Estimator newEstimator = template.getClass().getDeclaredConstructor().newInstance();
            
            // Set parameters if BaseEstimator
            if (newEstimator instanceof com.javaml.base.BaseEstimator) {
                ((com.javaml.base.BaseEstimator) newEstimator).setParams(params);
            }
            
            return newEstimator;
        } catch (Exception e) {
            // If we can't create a new instance, use the template
            if (template instanceof com.javaml.base.BaseEstimator) {
                ((com.javaml.base.BaseEstimator) template).setParams(params);
            }
            return template;
        }
    }
}

