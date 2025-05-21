# 📊 Linear Regression in C++

This repository contains C++ implementations of **Linear Regression**, showcasing two core approaches: an analytical method using statistics, and an iterative method using gradient descent.

---

## 🧠 Approaches

### 1. Statistical (Analytical) Approach
- File: `LR_statistical_aprch.cpp`
- Calculates the optimal **weight (slope)** and **bias (intercept)** directly using the formulas:
  - Sxx = ∑(xᵢ - x̄)²
  - Sxy = ∑(xᵢ - x̄)(yᵢ - ȳ)
- Derives weight and bias from:
  - `w = Sxy / Sxx`
  - `b = ȳ - w * x̄`
- Efficient for small to medium-sized datasets.

### 2. Gradient Descent Approach
- File: `LR_gradient_descent.cpp`
- Starts with random values for weight and bias.
- Iteratively updates parameters to minimize the cost function (Mean Squared Error):
  - `w -= learning_rate * ∂J/∂w`
  - `b -= learning_rate * ∂J/∂b`
- More flexible and scalable, useful for large datasets or complex extensions.

---
## ▶️ Run

```bash
g++ LR_statistical_aprch.cpp -o lr_stat && ./lr_stat
g++ LR_gradient_descent.cpp -o lr_gd && ./lr_gd
```
