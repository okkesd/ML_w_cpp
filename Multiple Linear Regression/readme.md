# C++ Linear Regression

An implementation of multivariate linear regression in C++ using the Eigen library for efficient matrix operations.

## Implementation Approach

The repository contains two approaches to linear regression:

### 1. Basic Implementation (`multi_LR.cpp`)
- Uses gradient descent with a combined bias term
- `fit_model()` function implements batch gradient descent:
  ```cpp
  // Main training loop
  for (int i = 0; i < max_iterations; i++) {
      // Calculate predictions (h(x) = X * theta)
      Eigen::VectorXd predictions = X * theta;
      
      // Calculate error (h(x) - y)
      Eigen::VectorXd errors = predictions - y;
      
      // Update parameters using gradient descent
      Eigen::VectorXd gradient = (X.transpose() * errors) / m;
      theta -= alpha * gradient;
      
      // Monitor cost function
      double cost = (errors.array().square().sum()) / (2.0 * m);
  }
  ```

### 2. Extended Implementation (`multi_seperate.cpp`)
- Separates bias term from feature weights
- Includes feature engineering and train/test splitting
- Provides evaluation metrics (MSE, R²)

## How to Use

```bash
# Compile and Run
g++ multi_LR.cpp -o multi_lr -I ../eigen-3.4.0 && ./multi_lr
g++ multi_seperate.cpp -o multi_seperate -I ../eigen-3.4.0/ && ./multi_seperate
```

## Key Features

- **Feature Normalization**: Standardizes features for faster convergence
- **Performance Metrics**: Calculates MSE and R² on test data
- **Feature Engineering**: Creates interaction terms between features

## Dependencies

- Eigen library (3.4.0+)
