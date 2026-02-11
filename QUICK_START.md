# JavaML Quick Start Guide

This guide demonstrates how to use JavaML for machine learning tasks.

## Basic Array Operations

```java
import com.javaml.array.*;

// Create arrays
NDArray arr1 = ArrayCreation.zeros(3, 3);
NDArray arr2 = ArrayCreation.ones(3, 3);
NDArray arr3 = ArrayCreation.arange(0, 10);

// Mathematical operations
NDArray result = ArrayMath.add(arr1, arr2);
NDArray sqrt = ArrayMath.sqrt(arr3);

// Linear algebra
NDArray matrix = ArrayCreation.randn(3, 3);
double det = LinearAlgebra.det(matrix);
NDArray product = LinearAlgebra.matmul(matrix, matrix.transpose());
```

## DataFrame Operations

```java
import com.javaml.dataframe.*;
import com.javaml.io.CSVReader;

// Create DataFrame
NDArray data = ArrayCreation.randn(100, 3);
DataFrame df = new DataFrame(data, "col1", "col2", "col3");

// Selection
DataFrame subset = df.loc(new Object[]{0, 1, 2}, "col1", "col2");
double value = df.at(0, "col1");

// GroupBy
DataFrameGroupBy grouped = DataFrameOperations.groupBy(df, "col1");
DataFrame aggregated = grouped.mean("col2");

// Merge
DataFrame df1 = new DataFrame(ArrayCreation.randn(50, 2), "id", "value1");
DataFrame df2 = new DataFrame(ArrayCreation.randn(50, 2), "id", "value2");
DataFrame merged = DataFrameOperations.merge(df1, df2, "id", "inner");
```

## Machine Learning Workflow

```java
import com.javaml.linear.LinearRegression;
import com.javaml.preprocessing.StandardScaler;
import com.javaml.model_selection.ModelSelection;

// 1. Prepare data
NDArray X = ArrayCreation.randn(100, 3);
NDArray y = createTarget(X);

// 2. Preprocess
StandardScaler scaler = new StandardScaler();
scaler.fit(X);
NDArray X_scaled = scaler.transform(X);

// 3. Split data
NDArray[] split = ModelSelection.trainTestSplit(X_scaled, y, 0.2);
NDArray X_train = split[0];
NDArray X_test = split[1];
NDArray y_train = split[2];
NDArray y_test = split[3];

// 4. Train model
LinearRegression model = new LinearRegression();
model.fit(X_train, y_train);

// 5. Evaluate
double score = model.score(X_test, y_test);
NDArray predictions = model.predict(X_test);
```

## I/O Operations

```java
import com.javaml.io.CSVReader;
import com.javaml.io.JSONReader;

// Read CSV
DataFrame df = CSVReader.readCSV("data.csv");

// Write CSV
CSVReader.toCSV(df, "output.csv");

// Read JSON
DataFrame df2 = JSONReader.readJSON("data.json");

// Write JSON
JSONReader.toJSON(df2, "output.json");
```

## Virtual Threads for Parallel Processing

```java
import com.javaml.util.VirtualThreadExecutor;
import java.util.function.Supplier;
import java.util.List;
import java.util.ArrayList;

VirtualThreadExecutor executor = new VirtualThreadExecutor();

// Execute multiple tasks in parallel
List<Supplier<Double>> tasks = new ArrayList<>();
for (int i = 0; i < 1000; i++) {
    final int idx = i;
    tasks.add(() -> {
        // Some computation
        return Math.sqrt(idx);
    });
}

List<Double> results = executor.executeAll(tasks);
```

## Complete Example

See `EndToEndExample.java` for a complete workflow demonstration.

## Next Steps

- Explore more preprocessing transformers (MinMaxScaler, LabelEncoder)
- Try different ML algorithms
- Use DataFrame operations for data manipulation
- Leverage virtual threads for parallel processing

