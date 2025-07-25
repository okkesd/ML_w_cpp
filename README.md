# ML_w_cpp

Machine learning applications implemented in C++ using Eigen library for linear algebra operations.

## Overview

This repository contains C++ implementations of fundamental machine learning algorithms and neural network architectures, built from scratch for educational and performance purposes.

## Features

- **Neural Networks**: Multi-layer perceptrons with backpropagation
- **Linear/Logistic Regression**: Gradient descent and statistical approaches
- **Attention Mechanisms**: Single-head attention implementation
- **Perceptron**: Basic linear classifier with training loop

## Structure

```
ML_w_cpp/
├── Linear_Regression/
│   ├── LR_gradient_descent.cpp
│   ├── LR_statistical_aprch.cpp
│   └── plot.py
├── Logistic_Regression/
│   └── Logistic_Regression.cpp
├── Multiple_Linear_Regression/
│   ├── multi_LR.cpp
│   └── multi_seperate.cpp
└── Neurons/
    ├── MultiLayerNN.cpp        # Multi-layer neural network
    ├── NNMultiOutput.cpp       # Multi-output neural network
    ├── SingleHead.cpp          # Single-head attention mechanism
    └── perceptron.cpp          # Basic perceptron
```

## Dependencies

- **Eigen3**: Linear algebra library
- **C++11** or later

## Building

```bash
# For neural networks (example)
g++ MultiLayerNN.cpp -o MultiLayerNN -I ../eigen-3.4.0/ && ./MultiLayerNN

# For perceptron
g++ perceptron.cpp -o perceptron -I ../eigen-3.4.0/ && ./perceptron
```

## Key Implementations

### Neural Networks
- **MultiLayerNN.cpp**: Feedforward network with sigmoid activation and backpropagation
- **NNMultiOutput.cpp**: Multi-output neural network with inheritance-based design
- **perceptron.cpp**: Single-layer perceptron with training convergence

### Regression
- **Linear Regression**: Both gradient descent and statistical approaches
- **Logistic Regression**: Binary classification with sigmoid function
- **Multiple Linear Regression**: Multivariate regression models

### Advanced Features
- **SingleHead.cpp**: Attention mechanism with scaled dot-product attention
- Custom activation functions (sigmoid)
- Matrix-based operations using Eigen

## Usage Example

```cpp
// Multi-layer neural network training
Eigen::MatrixXd data = prepare_input_data();
Eigen::VectorXd output(6);
output << 1,1,0,0,1,0;

InputLayer input_layer(3, 4);
HiddenLayer hidden_layer(4, 4);
OutputLayer output_layer(4, 1, 0);

NeuralNetwork nn(input_layer, hiddens, output_layer);
nn.train(data, output, 0.001, 1000);
```

## License

Open source - feel free to use and modify for educational purposes.
