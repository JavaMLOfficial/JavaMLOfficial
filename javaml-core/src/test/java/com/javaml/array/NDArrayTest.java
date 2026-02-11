package com.javaml.array;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic tests for NDArray to verify compilation and basic functionality.
 */
public class NDArrayTest {
    
    @Test
    public void testArrayCreation() {
        NDArray arr = ArrayCreation.zeros(3, 3);
        assertEquals(9, arr.getSize());
        assertEquals(2, arr.getNdims());
        assertArrayEquals(new int[]{3, 3}, arr.getShape());
    }
    
    @Test
    public void testArrayGetSet() {
        NDArray arr = ArrayCreation.zeros(2, 2);
        arr.set(5.0, 0, 0);
        assertEquals(5.0, arr.get(0, 0));
    }
    
    @Test
    public void testArange() {
        NDArray arr = ArrayCreation.arange(0, 5);
        assertEquals(5, arr.getSize());
        assertEquals(0.0, arr.get(0));
        assertEquals(4.0, arr.get(4));
    }
    
    @Test
    public void testOnes() {
        NDArray arr = ArrayCreation.ones(2, 2);
        assertEquals(1.0, arr.get(0, 0));
        assertEquals(1.0, arr.get(1, 1));
    }
}

