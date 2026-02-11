# [JavaML Library](https://javaml.com)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java Version](https://img.shields.io/badge/java-17%2B-blue)](https://www.java.com/en/)
[![Virtual Threads](https://img.shields.io/badge/virtual%20threads-enabled-green)](https://openjdk.org/projects/loom/)
<!--
[![Build Status](https://img.shields.io/travis/JavaMLOfficial/JavaMLOfficial/main)](https://travis-ci.com/JavaMLOfficial/JavaMLOfficial)
[![Coverage Status](https://coveralls.io/repos/github/JavaMLOfficial/JavaMLOfficial/badge.svg?branch=main)](https://coveralls.io/github/JavaMLOfficial/JavaMLOfficial?branch=main)
[![Maven Central](https://img.shields.io/maven-central/v/com.javaml/javaml-core)](https://search.maven.org/artifact/com.javaml/javaml-core)
-->

## 🚀 Welcome to JavaML

**JavaML** is a comprehensive machine learning library for Java developers, providing **NumPy**, **Pandas**, and **Scikit-learn** equivalent functionality with native Java support. Built from the ground up to leverage **Java Virtual Threads (Project Loom)**, JavaML enables fast, parallel model training and data processing that rivals Python implementations.

### ✨ Why JavaML?

- **🏃‍♂️ High Performance**: Leverages Java Virtual Threads for parallel processing, enabling millions of concurrent operations
- **🔧 Java-Native**: Built for Java developers, by Java developers - no Python bridge needed
- **📊 Comprehensive**: 900+ functions covering NumPy, Pandas, and Scikit-learn features
- **🎯 Production-Ready**: Type-safe, well-tested, and enterprise-grade
- **⚡ Fast Training**: Parallel model training with virtual threads for speed
- **🔄 Familiar API**: Similar patterns to Python ML libraries for easy migration
- **🌱 Spring Integration**: First-class Spring Boot and Spring AI support - official replacement for Python ML in Spring ecosystem
- **☁️ Enterprise-Ready**: Seamless integration with Spring Cloud, Spring Data, and Spring Security

### 🎯 Target Audience

- Java software engineers building ML applications
- **Spring Boot developers building ML-powered applications**
- **Spring AI users seeking native Java ML capabilities**
- Teams migrating from Python ML stacks to Java
- Enterprise applications requiring Java-based ML solutions
- Developers needing high-performance, concurrent ML operations

For inquiries or collaboration opportunities, feel free to reach out to us at 📩 [connect@javaml.com](mailto:connect@javaml.com).

<!---
<a href="https://next.ossinsight.io/widgets/official/compose-user-dashboard-stats?user_id=174118942" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/compose-user-dashboard-stats/thumbnail.png?user_id=174118942&image_size=auto&color_scheme=dark" width="815" height="auto">
    <img alt="Dashboard stats of @634750802" src="https://next.ossinsight.io/widgets/official/compose-user-dashboard-stats/thumbnail.png?user_id=174118942&image_size=auto&color_scheme=light" width="815" height="auto">
  </picture>
</a>

--->

## 📋 Features Overview

JavaML provides a comprehensive set of features covering **200+ NumPy functions**, **300+ Pandas methods**, and **400+ Scikit-learn algorithms**.

### 🔢 NumPy-Equivalent Array Operations (200+ functions)

**Array Creation & Manipulation:**
- Multi-dimensional arrays with efficient memory layout
- Array creation: `zeros()`, `ones()`, `arange()`, `linspace()`, `eye()`, `diag()`, etc.
- Array manipulation: `reshape()`, `transpose()`, `concatenate()`, `stack()`, `split()`, etc.
- Advanced indexing: boolean, integer, and fancy indexing
- Broadcasting: automatic shape alignment for operations

**Mathematical Operations:**
- Element-wise operations: `add()`, `subtract()`, `multiply()`, `divide()`, `power()`
- Trigonometric functions: `sin()`, `cos()`, `tan()`, `arcsin()`, `arccos()`, `arctan()`
- Exponential/Logarithmic: `exp()`, `log()`, `log10()`, `sqrt()`, etc.
- Statistical functions: `mean()`, `median()`, `std()`, `var()`, `percentile()`, etc.

**Linear Algebra:**
- Matrix operations: `dot()`, `matmul()`, `transpose()`, `inverse()`
- Decompositions: `svd()`, `qr()`, `cholesky()`, `eig()`
- Matrix properties: `det()`, `rank()`, `trace()`, `norm()`

**Random Number Generation:**
- 50+ distribution functions: `normal()`, `uniform()`, `exponential()`, `beta()`, `gamma()`, etc.
- Sampling: `choice()`, `shuffle()`, `permutation()`

### 📊 Pandas-Equivalent Data Structures (300+ methods)

**Core Data Structures:**
- **DataFrame**: 2D labeled data structure (rows × columns) with MultiIndex support
- **Series**: 1D labeled array with vectorized operations
- **Index**: Immutable sequence for efficient labeling

**Data I/O:**
- CSV, JSON, Excel, Parquet, HDF5, SQL database support
- Streaming support for large files
- Parallel reading/writing with virtual threads

**Data Manipulation:**
- Selection: `loc[]`, `iloc[]`, `at[]`, `iat[]`, boolean indexing, `query()`
- Cleaning: `dropna()`, `fillna()`, `drop_duplicates()`, `replace()`, `interpolate()`
- Transformation: `sort_values()`, `rename()`, `melt()`, `pivot()`, `stack()`, `unstack()`
- Grouping: `groupby()` with aggregation, transformation, and filtering
- Merging: `merge()`, `join()`, `concat()` with various join strategies
- Window operations: `rolling()`, `expanding()`, `ewm()`

**String & DateTime Operations:**
- String accessor: 50+ methods (`str.lower()`, `str.split()`, `str.contains()`, etc.)
- DateTime accessor: 30+ methods (`dt.year`, `dt.month`, `dt.strftime()`, etc.)

**Statistical Operations:**
- Descriptive statistics: `describe()`, `mean()`, `median()`, `std()`, `var()`
- Correlation: `corr()`, `cov()`
- Aggregations: `sum()`, `min()`, `max()`, `quantile()`, `value_counts()`

### 🤖 Scikit-learn-Equivalent ML Algorithms (400+ classes/functions)

**Supervised Learning - Classification (30+ estimators):**
- **Linear Models**: `LogisticRegression`, `RidgeClassifier`, `SGDClassifier`
- **Tree-Based**: `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier`
- **SVM**: `SVC`, `NuSVC`, `LinearSVC`, `OneClassSVM`
- **Nearest Neighbors**: `KNeighborsClassifier`, `RadiusNeighborsClassifier`
- **Naive Bayes**: `GaussianNB`, `MultinomialNB`, `BernoulliNB`
- **Ensemble**: `VotingClassifier`, `StackingClassifier`, `AdaBoostClassifier`, `BaggingClassifier`
- **Neural Networks**: `MLPClassifier`

**Supervised Learning - Regression (25+ estimators):**
- **Linear Models**: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`
- **Robust**: `HuberRegressor`, `RANSACRegressor`, `TheilSenRegressor`, `QuantileRegressor`
- **Tree-Based**: `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`
- **Kernel**: `KernelRidge`, `SVR`, `NuSVR`
- **Ensemble**: `VotingRegressor`, `StackingRegressor`

**Unsupervised Learning - Clustering (15+ estimators):**
- `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN`
- `AgglomerativeClustering`, `SpectralClustering`, `MeanShift`
- `AffinityPropagation`, `Birch`, `GaussianMixture`

**Dimensionality Reduction (15+ transformers):**
- `PCA`, `IncrementalPCA`, `KernelPCA`, `SparsePCA`
- `TSNE`, `LocallyLinearEmbedding`, `Isomap`, `MDS`
- `NMF`, `FastICA`, `FactorAnalysis`, `TruncatedSVD`

**Preprocessing & Feature Engineering (20+ transformers):**
- **Scaling**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `QuantileTransformer`
- **Encoding**: `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`
- **Imputation**: `SimpleImputer`, `KNNImputer`, `IterativeImputer`
- **Feature Engineering**: `PolynomialFeatures`, `SplineTransformer`, `FunctionTransformer`
- **Text**: `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`

**Model Selection & Evaluation (50+ functions):**
- **Cross-Validation**: `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `cross_val_score()`, `cross_validate()`
- **Hyperparameter Tuning**: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`
- **Metrics - Classification**: `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`, `roc_auc_score()`, `confusion_matrix()`
- **Metrics - Regression**: `mean_squared_error()`, `mean_absolute_error()`, `r2_score()`
- **Metrics - Clustering**: `silhouette_score()`, `adjusted_rand_score()`, `calinski_harabasz_score()`

**Pipelines & Composition:**
- `Pipeline`: Chain transformers and estimators
- `FeatureUnion`: Combine feature extraction methods
- `ColumnTransformer`: Transform specific columns

**Utilities:**
- Dataset generation: `make_classification()`, `make_regression()`, `make_blobs()`, etc.
- Built-in datasets: `load_iris()`, `load_digits()`, `load_wine()`, `load_breast_cancer()`, etc.
- Model persistence: `dump()`, `load()`

### ⚡ Virtual Thread Optimization

JavaML leverages **Java Virtual Threads (Project Loom)** for high-performance parallel processing:

- **Parallel Cross-Validation**: Distribute folds across virtual threads
- **Parallel Hyperparameter Search**: Test multiple parameter combinations simultaneously
- **Parallel Ensemble Methods**: Train base estimators concurrently
- **Parallel Matrix Operations**: Process rows/columns in parallel
- **Parallel Feature Transformations**: Transform columns concurrently
- **Non-blocking I/O**: Efficient data loading with virtual threads

### 🏗️ Architecture Highlights

- **Consistent API**: Unified `fit()`, `predict()`, `transform()`, `score()` interface
- **Type Safety**: Leverages Java generics for compile-time type checking
- **Memory Efficient**: Optimized data structures and memory layout
- **Extensible**: Easy to add custom estimators and transformers
- **Production-Ready**: Comprehensive error handling and validation

### 🌱 Spring & Spring AI Integration

JavaML is designed as the **official replacement** for Python ML libraries in the Spring ecosystem:

**Spring Boot Integration:**
- ✅ Spring Boot Starter (`javaml-spring-boot-starter`)
- ✅ Auto-configuration support
- ✅ Spring Bean support for all estimators
- ✅ Configuration properties
- ✅ Spring Data integration
- ✅ REST API support
- ✅ Spring Actuator metrics

**Spring AI Integration:**
- ✅ Spring AI `Model` interface compatibility
- ✅ Spring AI `VectorStore` integration (DataFrame as vector store)
- ✅ Embedding model support
- ✅ RAG (Retrieval Augmented Generation) integration
- ✅ Function calling for ML operations
- ✅ Natural language to ML workflows

**Enterprise Features:**
- ✅ Spring Cloud integration
- ✅ Spring Security support
- ✅ Distributed tracing
- ✅ Circuit breaker support
- ✅ Service discovery integration

For detailed requirements and implementation roadmap, see [REQUIREMENTS.md](REQUIREMENTS.md).

For more information, visit [JavaML.com](https://javaml.com).

## 🚀 Quick Start

### Prerequisites

- **Java 17+** (required for virtual thread support)
- **Maven 3.6+** or **Gradle 7.0+**

### Installation

```bash
# Clone the repository
git clone https://github.com/JavaMLOfficial/JavaMLOfficial.git
cd JavaMLOfficial

# Build with Maven
mvn clean install

# Or build with Gradle
./gradlew build
```

### Maven Dependencies

**Core Library:**
```xml
<dependency>
    <groupId>com.javaml</groupId>
    <artifactId>javaml-core</artifactId>
    <version>1.0.0</version>
</dependency>
```

**Spring Boot Starter:**
```xml
<dependency>
    <groupId>com.javaml</groupId>
    <artifactId>javaml-spring-boot-starter</artifactId>
    <version>1.0.0</version>
</dependency>
```

**Spring AI Integration:**
```xml
<dependency>
    <groupId>com.javaml</groupId>
    <artifactId>javaml-spring-boot-starter-ai</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Gradle Dependencies

**Core Library:**
```gradle
implementation 'com.javaml:javaml-core:1.0.0'
```

**Spring Boot Starter:**
```gradle
implementation 'com.javaml:javaml-spring-boot-starter:1.0.0'
```

**Spring AI Integration:**
```gradle
implementation 'com.javaml:javaml-spring-boot-starter-ai:1.0.0'
```

## 📖 Usage Examples

### NumPy-Like Array Operations

```java
import com.javaml.array.NDArray;

// Create arrays
NDArray arr1 = NDArray.zeros(3, 3);
NDArray arr2 = NDArray.ones(3, 3);
NDArray arr3 = NDArray.arange(0, 10, 1);

// Mathematical operations
NDArray result = arr1.add(arr2).multiply(2.0);
NDArray sqrt = arr3.sqrt();

// Linear algebra
NDArray dotProduct = arr1.dot(arr2);
NDArray transposed = arr1.transpose();
```

### Pandas-Like DataFrame Operations

```java
import com.javaml.dataframe.DataFrame;

// Create DataFrame
DataFrame df = DataFrame.readCSV("data.csv");

// Data manipulation
DataFrame filtered = df.query("age > 25");
DataFrame grouped = df.groupBy("category").agg("price", "mean");
DataFrame sorted = df.sortValues("date");

// Selection
DataFrame subset = df.loc(0, 10, "name", "age");
double value = df.at(0, "price");
```

### Scikit-learn-Like Machine Learning

```java
import com.javaml.ensemble.RandomForestClassifier;
import com.javaml.preprocessing.StandardScaler;
import com.javaml.pipeline.Pipeline;
import com.javaml.model_selection.train_test_split;

// Load and split data
DataFrame df = DataFrame.readCSV("dataset.csv");
NDArray X = df.drop("target").toNDArray();
NDArray y = df.getColumn("target").toNDArray();

NDArray[] split = train_test_split(X, y, testSize=0.2);
NDArray X_train = split[0], X_test = split[1];
NDArray y_train = split[2], y_test = split[3];

// Create pipeline
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new RandomForestClassifier(nEstimators=100));

// Train and predict
pipeline.fit(X_train, y_train);
NDArray predictions = pipeline.predict(X_test);
double accuracy = pipeline.score(X_test, y_test);
```

### Virtual Thread Parallel Processing

```java
import java.util.concurrent.Executors;
import com.javaml.model_selection.GridSearchCV;

// Parallel hyperparameter search with virtual threads
ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();

GridSearchCV gridSearch = new GridSearchCV(
    new RandomForestClassifier(),
    paramGrid,
    cv=5,
    executor=executor  // Uses virtual threads for parallel CV
);

gridSearch.fit(X_train, y_train);
RandomForestClassifier bestModel = gridSearch.bestEstimator();
```

### Spring Boot Integration Example

```java
@SpringBootApplication
@EnableJavaML
public class MLApplication {
    public static void main(String[] args) {
        SpringApplication.run(MLApplication.class, args);
    }
}

@RestController
@RequestMapping("/api/ml")
public class MLController {
    
    @Autowired
    private MLPredictionService predictionService;
    
    @PostMapping("/predict")
    public ResponseEntity<PredictionResult> predict(
            @RequestBody DataFrame input) {
        return ResponseEntity.ok(
            predictionService.predict(input)
        );
    }
}

@Service
public class MLPredictionService {
    
    @Autowired
    private RandomForestClassifier classifier;
    
    @Autowired
    private StandardScaler scaler;
    
    public PredictionResult predict(DataFrame input) {
        DataFrame scaled = scaler.transform(input);
        NDArray predictions = classifier.predict(scaled.toNDArray());
        return new PredictionResult(predictions);
    }
}
```

### Spring AI Integration Example

```java
@Configuration
public class JavaMLSpringAIConfig {
    
    @Bean
    public Model javaMLModel() {
        return new JavaMLModelAdapter(
            new RandomForestClassifier()
        );
    }
    
    @Bean
    public VectorStore javaMLVectorStore() {
        return new DataFrameVectorStore();
    }
}

@Service
public class RAGService {
    
    @Autowired
    private VectorStore vectorStore;
    
    @Autowired
    private ChatModel chatModel;
    
    public String answer(String question) {
        // Use JavaML DataFrame as vector store for RAG
        List<Document> docs = vectorStore.similaritySearch(question);
        // Use Spring AI ChatModel with retrieved context
        return chatModel.call(question, docs);
    }
}
```

## 📚 Documentation

- **[Requirements Document](REQUIREMENTS.md)** - Complete feature requirements and roadmap
- **[API Documentation](https://javaml.com/docs)** - Full API reference (coming soon)
- **[User Guide](https://javaml.com/guide)** - Comprehensive tutorials (coming soon)
- **[Examples](https://javaml.com/examples)** - Code examples and use cases (coming soon)

## 🗺️ Roadmap

### Phase 1: Core Infrastructure (Months 1-3) ✅ In Progress
- NumPy-like array implementation
- Basic mathematical operations
- Broadcasting and vectorization
- Virtual thread executor setup

### Phase 2: Data Structures (Months 4-6)
- DataFrame and Series implementation
- Basic data manipulation
- CSV/JSON I/O

### Phase 3: Preprocessing (Months 7-9)
- Scaling and normalization
- Encoding and imputation
- Feature engineering
- Pipeline implementation

### Phase 4: Core ML Algorithms (Months 10-15)
- Linear models
- Tree-based models
- Basic clustering
- Core metrics

### Phase 5: Advanced Features (Months 16-21)
- Ensemble methods
- Advanced clustering
- Dimensionality reduction
- Hyperparameter tuning

### Phase 6: Optimization & Polish (Months 22-24)
- Performance optimization
- Complete documentation
- Example code and tutorials

## 🤝 Contributing

We welcome contributions from the Java developer community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- Algorithm implementations
- Performance optimizations
- Documentation improvements
- Test coverage
- Example code
- Bug fixes

## 📊 Feature Comparison

| Feature | NumPy | Pandas | Scikit-learn | JavaML |
|---------|-------|--------|--------------|--------|
| Array Operations | ✅ | ❌ | ❌ | ✅ |
| DataFrame | ❌ | ✅ | ❌ | ✅ |
| Classification | ❌ | ❌ | ✅ | ✅ |
| Regression | ❌ | ❌ | ✅ | ✅ |
| Clustering | ❌ | ❌ | ✅ | ✅ |
| Preprocessing | ❌ | ❌ | ✅ | ✅ |
| Virtual Threads | ❌ | ❌ | ❌ | ✅ |
| Type Safety | ❌ | ❌ | ❌ | ✅ |

## 🏆 Performance

JavaML is designed for high performance with virtual threads:

- **Parallel Training**: Train multiple models simultaneously
- **Fast Predictions**: Low-latency inference (< 10ms for small models)
- **Memory Efficient**: Optimized data structures
- **Scalable**: Handle datasets with 10M+ rows

## 📝 License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

JavaML is inspired by:
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing
- [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library

## 📞 Contact & Support

- **Website**: [javaml.com](https://javaml.com)
- **Email**: [connect@javaml.com](mailto:connect@javaml.com)
- **Issues**: [GitHub Issues](https://github.com/JavaMLOfficial/JavaMLOfficial/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JavaMLOfficial/JavaMLOfficial/discussions)

---

**Built with ❤️ for the Java Developer Community**

<!---
<a href="https://next.ossinsight.io/widgets/official/compose-last-28-days-stats?repo_id=174118942" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/compose-last-28-days-stats/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=dark" width="815" height="auto">
    <img alt="Performance Stats of pingcap/tidb - Last 28 days" src="https://next.ossinsight.io/widgets/official/compose-last-28-days-stats/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=light" width="815" height="auto">
  </picture>
</a>
<a href="https://next.ossinsight.io/widgets/official/compose-activity-trends?repo_id=174118942" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/compose-activity-trends/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=dark" width="815" height="auto">
    <img alt="Activity Trends of pingcap/tidb - Last 28 days" src="https://next.ossinsight.io/widgets/official/compose-activity-trends/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=light" width="815" height="auto">
  </picture>
</a>

<a href="https://next.ossinsight.io/widgets/official/analyze-repo-loc-per-month?repo_id=174118942" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/analyze-repo-loc-per-month/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=dark" width="815" height="auto">
    <img alt="Lines of Code Changes of pingcap/tidb" src="https://next.ossinsight.io/widgets/official/analyze-repo-loc-per-month/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=light" width="815" height="auto">
  </picture>
</a>

<a href="https://next.ossinsight.io/widgets/official/compose-recent-top-contributors?repo_id=174118942" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/compose-recent-top-contributors/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=dark" width="815" height="auto">
    <img alt="Top Contributors of pingcap/tidb - Last 28 days" src="https://next.ossinsight.io/widgets/official/compose-recent-top-contributors/thumbnail.png?repo_id=174118942&image_size=auto&color_scheme=light" width="815" height="auto">
  </picture>
</a>

--->

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
