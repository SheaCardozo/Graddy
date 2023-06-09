# Graddy
Graddy is a basic ML framework implemented from scratch in Python. It comes with an implementation of AutoGrad, data structures for Tensors and Modules, basic deep learning layers such as Linear and Convolutional layers with ReLU and SoftMax activations, and optimizer classes for SGD and SGD with Momentum. It should be fairly simple to extend the existing classes to further custom layers.

This originally started as an experiment in building a simple variation of AutoGrad from the ground up - inspired by Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd) - but spiraled out into a much larger project. Graddy is capable of building, training and evaluating basic classification models such as a convolutional neural network trained on the MNIST classification task. 
