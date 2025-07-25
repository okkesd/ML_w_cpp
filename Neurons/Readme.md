# Neurons

Neural network implementations in C++ using Eigen for matrix operations.

## Files

### Core Neural Networks
- **`MultiLayerNN.cpp`** - Multi-layer neural network with backpropagation training
- **`NNMultiOutput.cpp`** - Multi-output neural network with inheritance-based layer design
- **`perceptron.cpp`** - Basic single-layer perceptron with convergence training

### Advanced Components
- **`SingleHead.cpp`** - Single-head attention mechanism with scaled dot-product attention (incomplete transformer block)

## Features

- **Activation Functions**: Sigmoid, softmax
- **Training**: Stochastic gradient descent with backpropagation
- **Architecture**: Configurable hidden layers and output neurons
- **Matrix Operations**: Efficient computation using Eigen library

## Quick Build & Run

```bash
# Multi-layer network
g++ MultiLayerNN.cpp -o MultiLayerNN -I ../eigen-3.4.0/ && ./MultiLayerNN

# Multi-output network  
g++ NNMultiOutput.cpp -o NNMultiOutput -I ../eigen-3.4.0/ && ./NNMultiOutput

# Perceptron
g++ perceptron.cpp -o perceptron -I ../eigen-3.4.0/ && ./perceptron
```

## Network Architectures

| File | Input | Hidden | Output | Features |
|------|-------|--------|--------|----------|
| MultiLayerNN | 3 neurons | 2x4 layers | 1 neuron | Single output classification |
| NNMultiOutput | 3 neurons | 3 layers (4,5,6) | 2 neurons | Multi-output with inheritance |
| perceptron | 3 neurons | None | 1 neuron | Linear classification |

All implementations include training data preparation and demonstrate basic ML workflows.
