package com.javaml.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * Utility class for managing virtual thread executors.
 * Provides a centralized way to create and manage virtual thread executors
 * for parallel operations in JavaML.
 * 
 * <p>Virtual threads (Project Loom) enable millions of concurrent operations
 * with minimal overhead, making them ideal for parallel ML operations like
 * cross-validation, hyperparameter search, and ensemble training.</p>
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
public class VirtualThreadExecutor {
    
    private final ExecutorService executor;
    private final boolean isOwned;
    
    /**
     * Creates a new VirtualThreadExecutor with a default virtual thread executor.
     */
    public VirtualThreadExecutor() {
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        this.isOwned = true;
    }
    
    /**
     * Creates a new VirtualThreadExecutor with the provided executor.
     * 
     * @param executor the executor to use
     */
    public VirtualThreadExecutor(ExecutorService executor) {
        if (executor == null) {
            throw new IllegalArgumentException("Executor cannot be null");
        }
        this.executor = executor;
        this.isOwned = false;
    }
    
    /**
     * Gets the underlying executor service.
     * 
     * @return the executor service
     */
    public ExecutorService getExecutor() {
        return executor;
    }
    
    /**
     * Submits a task for execution.
     * 
     * @param task the task to execute
     * @param <T> the result type
     * @return a Future representing the result
     */
    public <T> Future<T> submit(Supplier<T> task) {
        return executor.submit(() -> task.get());
    }
    
    /**
     * Submits a runnable task for execution.
     * 
     * @param task the task to execute
     * @return a Future representing the completion
     */
    public Future<?> submit(Runnable task) {
        return executor.submit(task);
    }
    
    /**
     * Executes multiple tasks in parallel and collects results.
     * 
     * @param tasks the tasks to execute
     * @param <T> the result type
     * @return a list of results
     * @throws InterruptedException if interrupted while waiting
     */
    public <T> List<T> executeAll(List<Supplier<T>> tasks) throws InterruptedException {
        List<Future<T>> futures = new ArrayList<>();
        
        for (Supplier<T> task : tasks) {
            futures.add(submit(task));
        }
        
        List<T> results = new ArrayList<>();
        for (Future<T> future : futures) {
            try {
                results.add(future.get());
            } catch (java.util.concurrent.ExecutionException e) {
                throw new RuntimeException("Task execution failed", e.getCause());
            }
        }
        
        return results;
    }
    
    /**
     * Executes multiple tasks in parallel using parallel streams.
     * 
     * @param tasks the tasks to execute
     * @param <T> the result type
     * @return a list of results
     */
    public <T> List<T> executeAllParallel(List<Supplier<T>> tasks) {
        return tasks.parallelStream()
            .map(Supplier::get)
            .collect(Collectors.toList());
    }
    
    /**
     * Shuts down the executor gracefully.
     * If this executor was created by this instance, it will be shut down.
     * Otherwise, it will not be shut down (caller's responsibility).
     * 
     * @throws InterruptedException if interrupted while waiting
     */
    public void shutdown() throws InterruptedException {
        if (isOwned) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                throw e;
            }
        }
    }
    
    /**
     * Shuts down the executor immediately.
     * 
     * @return list of tasks that were awaiting execution
     */
    public List<Runnable> shutdownNow() {
        if (isOwned) {
            return executor.shutdownNow();
        }
        return List.of();
    }
    
    /**
     * Checks if the executor is shut down.
     * 
     * @return true if shut down
     */
    public boolean isShutdown() {
        return executor.isShutdown();
    }
    
    /**
     * Checks if the executor is terminated.
     * 
     * @return true if terminated
     */
    public boolean isTerminated() {
        return executor.isTerminated();
    }
    
    /**
     * Gets a default shared instance of VirtualThreadExecutor.
     * This instance should not be shut down as it's shared.
     * 
     * @return a shared VirtualThreadExecutor instance
     */
    public static VirtualThreadExecutor getDefault() {
        return DefaultInstanceHolder.INSTANCE;
    }
    
    /**
     * Holder for the default shared instance.
     */
    private static class DefaultInstanceHolder {
        private static final VirtualThreadExecutor INSTANCE = new VirtualThreadExecutor();
    }
}

