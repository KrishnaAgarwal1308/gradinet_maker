from plot_graph import draw_dot
import math
class Values:
    def __init__(self, data, _children=(), _op = "", label = "" ):#label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
    def __repr__(self):
        return f"the value data is: {self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Values) else Values(other)
        out = Values(self.data + other.data, (self, other),"+")
        def _backward():
            self.grad +=  1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Values) else Values(other)
        out = Values(self.data * other.data, (self, other),"*")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward 
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) -1) / (math.exp(2*x)+1)
        out = Values(t, (self,), "tanh")
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data 
        out = Values(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def __pow__(self,other):
        assert isinstance(other, (int, float))
        out = Values(self.data**other, (self,), "pow")

        def _backward ():
            self.grad += (other * self.data **(other-1) )* out.grad
        out._backward = _backward
        return out 

    def relu (self):
        x = self.data
        t = x if x>0 else 0.05*x
        out = Values(t, (self,), "relu")

        def _backward ():
            self.grad += out.grad * (1 if x>0 else 0)

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data 
        t = 1/(1+math.exp(-x))
        out = Values(t, (self,),"sigmoid")
        def _backward():
            self.grad += out.grad * (t*(1-t))
        out._backward = _backward
        return out
    
    def swish(self):
        x = self.data
        sigmoid = 1/(1+math.exp(-x))
        t = x * sigmoid
        out = Values(t, (self,),'swish')
        def _backward():
            self.grad += out.grad * (sigmoid + t*(1-sigmoid))
        out._backward = _backward
        return out


    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self,other):
        return other + (-self)
    def __neg__(self):
        return self * -1 
    def __truediv__(self, other):
        return (self * other **-1)
    def __rmul__(self,other):
        return self * other
    



    def backward(self):
        topo_order = []
        visited = set()
        def topo(v):
            if v not in visited :
                visited.add(v)
                for node in v._prev:
                    topo(node)
                topo_order.append(v)
        topo(self)
        self.grad = 1
        for nodes in reversed(topo_order):
            nodes._backward()
            



# inputs x1,x2
# x1 = Values(2.0, label='x1')
# x2 = Values(0.0, label='x2')
# # weights w1,w2
# w1 = Values(-3.0, label='w1')
# w2 = Values(1.0, label='w2')
# # bias of the neuron
# b = Values(6.8813735870195432, label='b')
# # x1*w1 + x2*w2 + b
# x1w1 = x1*w1; x1w1.label = 'x1*w1'
# x2w2 = x2*w2; x2w2.label = 'x2*w2'
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# n = x1w1x2w2 + b; n.label = 'n'
# o = n.tanh(); o.label = 'o'

# o.backward()


