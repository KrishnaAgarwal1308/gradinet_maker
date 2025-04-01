from gradient_maker import Values
import random
import torch
import math

class Neuron:
    def __init__(self, nins):
        self.w = [Values(random.uniform(-1,1)) for _ in range (nins)]
        self.b = Values(0)
    
    def __call__(self, x, **kwds):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        # print(f"Activation before ReLU: {act.data}")  # Debug
        return act.tanh()

    def parameters(slef):
        return slef.w+[slef.b]
    

class Layer: 
    def __init__ (self, nins, nouts):
        self.neurons = [Neuron(nins) for _ in range (nouts)]
    
    def __call__(self, x, **kwds):
        out = [n(x) for n in self.neurons]
        return out
    def parameters (self):
        return [ p for n in self.neurons for p in n.parameters()]

class Mlp:
    def __init__(self, nins, nouts):
        size = [nins] + nouts
        self.layers = [Layer(size[i],size[i+1])for i in range(len(nouts))]
    
    def __call__(self, x, **kwds):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [ p for n in self.layers for p in n.parameters()]
        
# Define the MLP model
n = Mlp(6, [4, 4, 1])

# Input dataset
xs = [[-1, 0, 1, 1, -1, 1],
      [-1, 0, 1, -1, 1, 1],
      [-1, 1, 1, 1, -1, 1],
      [-1, 1, 1, 0, -1, -1],
      [-1, -1, 1, 1, -1, 1],
      [-1, 0, .1, 1, -1, 0,1],
      [-1, 0.5, 1, -1, -.1, 1],
      [-.2, 0, 1, .1, -1, 1],
      [-1, 0, 1, 1, -1, 1],
      [-1, 0, 1, 1, -1, 1]]

ys = [1, -1, 0, -1, 0, 1, -1, 1, 1, 0]

# Training Loop

# Warm-up runs
warmup_steps = 100
warmup_lr = 0.001  # Lower learning rate for warm-up

for i in range(warmup_steps):
    ypred = [n(x)[0] for x in xs]
    
    # Compute loss
    loss = sum((yp - y) ** 2 for yp, y in zip(ypred, ys))
    
    # Reset gradients
    for p in n.parameters():
        p.grad = 0.0
    
    # Backpropagation
    loss.backward()

    # Gradient Descent Update with warm-up learning rate
    for p in n.parameters():
        p.data += -warmup_lr * p.grad
    
    if i % 20 == 0:  # Print progress every 20 steps
        print(f"Warm-up Step: {i}, Loss: {loss.data}")

# Main Training Loop
learning_rate = 0.1
for i in range(5000):
    ypred = [n(x)[0] for x in xs]
    
    # Compute loss
    loss = sum((yp - y) ** 2 for yp, y in zip(ypred, ys))
    
    # Reset gradients
    for p in n.parameters():
        p.grad = 0.0
    
    # Backpropagation
    loss.backward()

    # Gradient Descent Update with main learning rate
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    if i % 100 == 0:
        print(f"Step: {i}, Loss: {loss.data}")

print("Final Predictions:", ypred)
