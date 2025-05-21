# C++ Logistic Regression Implementation

A binary classification model implemented in C++ using the Eigen library for efficient matrix operations.

## Mathematical Implementation

This implementation focuses on the core mathematical functions of logistic regression:

### 1. Sigmoid Function

The sigmoid function maps any real value to a probability between 0 and 1:

```cpp
double sigmoid_func(double input) {
    return 1.0 / (1.0 + exp(-input));
}
```

### 2. Loss Function

The implementation uses binary cross-entropy loss for binary classification:

```cpp
// For a single sample:
double loss = y == 1 ? -log(prediction) : -log(1.0 - prediction);

// Vectorized implementation for all samples:
Eigen::VectorXd loss(predictions.size());
for (int i = 0; i < loss.size(); i++) {
    loss(i) = y(i) == 1 ? -log(predictions(i)) : -log(1.0 - predictions(i));
}
```

### 3. Cost Calculation

The cost is calculated as the average loss across all samples:

```cpp
double current_cost = loss.array().sum() / loss.size();
```

### 4. Gradient Descent Update

Parameters are updated using the gradient of the cost function:

```cpp
// Update weight parameters
for (int i = 0; i < X.cols(); i++) {
    theta[i] -= learning_rate * (((predictions - y).transpose() * X.col(i)).sum() / X.rows());
}

// Update bias term
constant -= learning_rate * ((predictions - y).sum() / X.rows());
```

## Key Features

- **Categorical Feature Handling**:
  - Binary encoding for gender and previous loan file
  - One-hot encoding for home ownership and loan intent
  - Ordinal encoding for education level

- **Feature Engineering & Preprocessing**:
  - Feature normalization for numeric columns
  - Handling of mixed data types

- **Model Evaluation**:
  - k-fold cross-validation 
  - Train/test splitting

## Usage

```bash
# Compile and Run
g++ Logistic_Regression.cpp -o logistic -I path/to/eigen-3.4.0/ && ./logistic
```

## Dependencies

- Eigen library (3.4.0+)
