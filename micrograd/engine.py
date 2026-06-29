from __future__ import annotations
import numpy 
import matplotlib
import math


class Value:
    _ops = {
        '+': lambda a,b : a + b,
        '-': lambda a,b : a - b,
        '*': lambda a,b : a * b,
        '/': lambda a,b : a / b,
    }
    _grad_ops = {
        '+': lambda a,b : (1.0,1.0),
        '*': lambda a,b : (b,a),
        '-': lambda a,b : (1.0,-1.0),
        '/': lambda a,b : (1.0/b, -a/(b**2)),
        'exp': lambda a,b : (a),
        'tanh': lambda a,b : (1-a^2)
        
    }
    def __init__(self,data,_children = (),_op = '',label = ''):
        self.data = data
        self.grad = 0.0 #stores derivative of Value WRT some final function L
        self._prev = set(_children)
        self._op = _op #stores the operation as a str that resulted in this Value
        self._nodegraph = ()
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        #If a value object is called, python uses __repr__ implicitly. eg a = Value(3.0) print(a) would call __repr__
        return f"Value(data = {self.data})"
    
    def __add__(self,other)->Value:
        #Called implicitly during a + b where a and b are Value functions, and python calls a.__add__(b). A Value object is returned.
        other = other if isinstance(other,Value) else Value(other) #create value fn to allow for Value(a) + b operations
        out = Value(self.data + other.data,(self,other),"+")
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out #note: the return type stores it's children. This creates a computational graph to be followed recursively

    def __radd__(self, other): # other + self
        return self + other
    def __sub__(self,other)->Value:
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data - other.data,(self,other),"-")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = self * other**-1.0
        def _backward():
            self.grad += out.grad * other.data**-1.0
            other.grad += out.grad * (-self.data * other.data**-2.0)

        out._backward = _backward
        return out


    def __pow__(self,k):
        out = Value(self.data**k, (self, ), _op = "pow")
        def _backward():
            self.grad += (k * self.data ** (k-1)) * out.grad
        out._backward = _backward
        return out


    def __mul__(self,other)->Value:
        other = other if isinstance(other,Value) else Value(other)
        #Called implicitly during a * b where a and b are Value functions, and python calls a.__mul__(b). A Value object is returned.
        out = Value(self.data * other.data,(self,other),"*")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        #Called implicitly during a * b where a and b are Value functions, and python calls a.__mul__(b). A Value object is returned.
        out = Value(other.data * self.data,(other,self),"*")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    
    def exp(self):
        out = Value(math.exp(self.data),(self, ),_op = 'exp')
        def _backward():
            self.grad = out.grad * out.data
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data),(self, ),_op = 'tanh')
        def _backward():
            self.grad += (1-out.data**2)*out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(0,self.data),(self, ), _op = 'relu')
        def _backward():
            grad = 1 if out.data > 1 else 0
            self.grad += out.grad * grad
        out._backward = _backward
        return out
    

    
    def backprop(self):
        #build topological graph then call the backward pass in reverse order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    
    
            



# x1 = Value(2.0,label = "x1")
# x2 = Value(0.0, label = "x2" )
# w1 = Value(-3.0, label = "w1")
# w2 = Value(1.0, label = "w2")

# b = Value(6.881373587,label = "b")
# print(w1/x1)
# x1w1 = x1 * w1; x1w1.label = "x1w1"
# x2w2 = x2 * w2; x2w2.label = "x2w2"

# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1 + x2w2"

# n = x1w1x2w2 + b; n.label = "n"
# o = n.tanh(); o.label = "o"

# # inter = (n*2).exp(); inter.label = "inter"
# # o = (inter - 1.0)/(inter + 1.0)

# o.backprop()
# o.draw_nodegraph()

   

