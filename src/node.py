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
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_bias = np.zeros(self.bias.shape)
        self.prev_input = np.zeros(self.weights.shape)


    def forward(self, input):
        self.prev_input = input
        return np.inner(self.weights, input) + self.bias

    def backward(self, loss):
        self.grad_bias = np.array([1.])
        self.grad_weights = self.prev_input
        return self.weights
    
    def update(self, step_fn):
        self.bias = self.bias - step_fn(self.grad_bias)
        self.weights = self.weights - step_fn(self.grad_weights)
    

class Linear:
    def __init__(self, num_inputs, num_outputs):
        self.layer = Stack([LinearNode(num_inputs) for _ in range(num_outputs)])

    def forward(self, input):
        return self.layer.forward(input)[:, 0]

    def backward(self, loss):
        return self.layer.backward(loss)
    
    def update(self, step_fn):
        self.layer.update(step_fn)
    



class Stack:
    def __init__(self, modules):
        self.modules = modules
    
    def forward(self, input):
        output_list = []
        for m in self.modules:
            output_list.append(m.forward(input))
        return np.stack(output_list)
    
    def backward(self, loss):
        back_loss = None
        for m in self.modules:
            if back_loss is None:
                back_loss = m.backward(loss)
            else :
                back_loss += m.backward(loss)
        return back_loss
    
    def update(self, step_fn):
        for m in self.modules:
            m.update(step_fn)
        
class Sequence:

    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input

    def backward(self, loss):
        for l in reversed(self.layers):
            loss = l.backward(loss)
        return loss
    
    def update(self, step_fn):
        for l in self.layers:
            l.update(step_fn)
        

class Sigmoid:

    def __init__(self):
        pass

    def forward(self, input):
        return 1./(1+np.exp(-input))
    
    def backward(self, loss):
        return 1./(1+np.exp(-loss))*1./(1+np.exp(loss))
    
    def update(self, step_fn):
        return


class SGD:

    def __init__(self, model, lr = 1e-3):
        self.lr = lr
        self.model = model
    
    def update_parameters(self, loss):
        self.model.backward(loss)
        self.model.update(self.get_step)

    def get_step(self, grad):
        return self.lr * grad

l = Sequence([Linear(2, 4), Linear(4, 3), Linear(3, 1), Sigmoid()])
l.forward(np.array([1., 4.]))
l.backward(np.array([1.]))

sgd = SGD(l)

sgd.update_parameters(np.array([-2.]))


