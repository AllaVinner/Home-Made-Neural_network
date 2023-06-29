from typing import Protocol
import numpy as np


class LinearNode:
    def __init__(self, num_inputs: int, initial_weights = None, initial_bias = None):
        if initial_weights is None:
            initial_weights = np.random.randn(num_inputs)    
        if initial_bias is None:
            initial_bias = np.random.randn(1)

        self.weights = initial_weights
        self.bias = initial_bias


    def forward(self, input):
        return np.inner(self.weights, input) + self.bias


class Linear:
    def __init__(self, num_inputs, num_outputs):
        self.layer = Stack([LinearNode(num_inputs) for _ in range(num_outputs)])

    def forward(self, input):
        return self.layer.forward(input)

class Stack:
    def __init__(self, modules):
        self.modules = modules
    
    def forward(self, input):
        output_list = []
        for m in self.modules:
            output_list.append(m.forward(input))
        return np.stack(output_list)

class Sequence:

    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input


l = Linear(2, 4)

l.forward(np.array([1., 4.]))

