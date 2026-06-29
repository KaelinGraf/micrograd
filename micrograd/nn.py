from __future__ import annotations
from engine import Value as V
from draw import draw_nodegraph
import random

class Neuron:
    def __init__(self,nin):
        #nin is number of inputs to the neuron
        self.w = [V(random.uniform(-1,1))for _ in range(nin)] #create nin input weights
        self.b = V(random.uniform(-1,1))

    def __call__(self,x):
        #performs w * x + b
        act = sum((wi * xi for wi,xi in zip(self.w,x)),self.b) 
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]




class FullyConnectedLayer:
    def __init__(self,nin,nout):
        self.n = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.n]
        return outs
    
    def parameters(self):
        return[p for n in self.n for p in n.parameters()]


class MLP:
    def __init__(self,nin,nouts,step_size=0.001):
        #note: nouts is a list of mlp layers
        self.step_size = step_size
        self.sz = [nin] + nouts #append lists
        self.layers = [FullyConnectedLayer(self.sz[i],self.sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        #for inputs x
        if len(x) != self.sz[0]:raise Exception(f"Input must be exactly length {self.sz[0]}, provided {len(x)}")
        for layer in self.layers:
            x = layer(x) #pass in vector x
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def update(self):
        for p in self.parameters():
            p.data += p.grad * -self.step_size

    
mlp = MLP(3,[4,4,1],step_size=0.05)

xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]
ys = [1.0,-1.0,-1.0,1.0]


iters = 100
loss_list = []
for i in range(iters):
    ypred = [mlp(x)[0] for x in xs]
    #IMPLIMENTING BASIC CROSS ENTROPY LOSS
    loss= sum((yout-ygt)**2 for ygt,yout in zip(ys,ypred))
    loss_list.append(loss.data)
    for p in mlp.parameters():p.grad=0.0
    loss.backprop()
    # draw_nodegraph(loss)
    mlp.update()


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,iters,1)
plt.plot(x,loss_list)
plt.show()