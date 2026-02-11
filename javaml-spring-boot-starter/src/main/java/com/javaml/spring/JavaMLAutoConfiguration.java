package com.javaml.spring;

import com.javaml.util.VirtualThreadExecutor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Spring Boot auto-configuration for JavaML.
 * Provides default beans for JavaML components.
 * 
 * @author JavaML Development Team
 * @version 1.0.0
 */
@Configuration
public class JavaMLAutoConfiguration {
    
    /**
     * Creates a VirtualThreadExecutor bean if not already present.
     * 
     * @return a VirtualThreadExecutor instance
     */
    @Bean
    @ConditionalOnMissingBean
    public VirtualThreadExecutor virtualThreadExecutor() {
        return VirtualThreadExecutor.getDefault();
    }
}

