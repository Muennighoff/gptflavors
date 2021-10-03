import math
import random

### Micrograd Framework https://github.com/karpathy/micrograd ###

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    ### Added from a micrograd pr ###
    def exp(self):
        try:
            out = Value(math.exp(self.data), (self,), "e")
        except:
            # This happens when e.g. not using Layernorm
            print("Data too large: ", self.data)

        def _backward():
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward

        return out

    ### Added ###
    def log(self):

        out = Value(math.log(self.data), (self,), "ln")

        def _backward():
            self.grad += (1 / self.data) * out.grad
        
        out._backward = _backward
        
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

### HELPER FUNCTIONS ###

def dot(v1, v2):
    """Vector dot product of two vectors with same length"""
    return sum([x*y for x,y in zip(v1, v2)])

def matmul(A, B):
    """
    Args:
      A: Matrix with shape (X, Y) = X rows of length Y
      B: Matrix with shape (Y, Z) = Y rows of length Z = Z cols of length Y

    Returns:
      Matrix with shape (X, Z)
    """
    # Invert B to be (Z, Y)
    B = list(zip(*B))
    out = []
    for row_a in A:
        new_row = []
        for col_b in B:
            # row of length Y & col of length Y
            assert len(row_a) == len(col_b)
            new_row.append(dot(row_a, col_b))
        out.append(new_row)
    return out

def layernorm(x, eps=1e-5):
    """Calculates simplified layernorm over rows of 2-dim array"""

    means = [sum(row) / len(row) for row in x]

    # Calculate variance
    vars = []
    for i in range(len(x)):
        mean_diffs = []
        for j in range(len(x[i])):
            mean_diffs.append((x[i][j] - means[i]) ** 2)
        vars.append(sum(mean_diffs) / len(mean_diffs))

    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = (x[i][j] - means[i]) / ((vars[i] + eps) ** (1/2))

    return x


def softmax(x, dim=1):
    """Applies softmax to 2-dimensional array - This can be done better"""
    if dim == 0:
        x = list(zip(*x))
    
    out = []
    for i in range(0, len(x)):
        out.append([x[i][j].exp() for j in range(0, len(x[i]))])

    for i in range(0, len(x)):
        denom = sum(out[i])
        for j in range(len(out[i])):
            out[i][j] /= denom

    if dim == 0:
        return list(zip(*out))
    return out



### Core GPT Modules ###

### Glossary
# B = Batch size
# D = Hidden dim
# H = num Heads
# S = Sequence length = block Size
# L = Layers
# V = Vocab size
# E = Embedding dim
# P = dropout Probability

class SelfAttention(Module):
    def __init__(self, D, S):
        super().__init__()

        # Micrograd doesn't give an option to turn off bias
        self.q_dense = Layer(D, D)
        self.k_dense = Layer(D, D)
        self.v_dense = Layer(D, D)

        # Micrograd has no dropout

    def __call__(self, x):
        
        # No batch size 
        S = len(x)
        D = len(x[0])

        # Apply the same linear weights for each seq length unit
        q = list(map(self.q_dense, x)) # Shape: [S, D]
        k = list(map(self.k_dense, x)) # Shape: [S, D]
        v = list(map(self.v_dense, x)) # Shape: [S, D]

        out = matmul(q, list(zip(*k))) # Shape: [S, S]

        # Masking of future values
        for i in range(0, S):
            for j in range(i+1, S):
                # Hide the ith row * jth column
                out[i][j].data = -9999 # using float('-inf') introduces NaNs

        # Apply softmax across columns of each row
        out = softmax(out, dim=1) # Shape: [S, S]

        out = matmul(out, v) # Shape: [S, D]

        return out

    def parameters(self):
        q_pars = [p for p in self.q_dense.parameters()]
        k_pars = [p for p in self.k_dense.parameters()]
        v_pars = [p for p in self.v_dense.parameters()]
        return q_pars + k_pars + v_pars


class GPTBlock(Module):
    def __init__(self, D, S):
        super().__init__()
        
        self.att = SelfAttention(D, S)
        
        # MLP without dropout
        self.ff = MLP(D, [D*4, D*4, D])

    def __call__(self, x):

        att_out = self.att(layernorm(x))

        # Add Residual
        for i in range(len(att_out)):
            for j in range(len(att_out[i])):
                att_out[i][j] += x[i][j]

        ff_out = list(map(self.ff, layernorm(att_out)))

        # Add Residual
        for i in range(len(ff_out)):
            for j in range(len(ff_out[i])):
                ff_out[i][j] += att_out[i][j]

        return ff_out

    def parameters(self):
        att_pars = self.att.parameters()
        ff_pars = self.ff.parameters()
        return att_pars + ff_pars


class GPT(Module):
    def __init__(self, L=2, D=8, S=12, V=26, P=0.2):
        super().__init__()

        # Embeddings
        self.voc_emb = [[Value(random.uniform(-1,1)) for _ in range(D)] for _ in range(V)]
        self.pos_emb = [[Value(random.uniform(-1,1)) for _ in range(D)] for _ in range(S)]

        # Transformer Blocks
        self.blocks = [GPTBlock(D, S) for _ in range(L)]

        # Output
        self.out = Layer(D, V)

    def __call__(self, x):
        """
          Args:
            x: List of length S
          Returns:
            y: List of length S
        """
        voc_emb = [self.voc_emb[idx] for idx in x] # Shape [S, D]
        pos_emb = [self.pos_emb[idx] for idx in range(len(x))] # Shape [S, D]

        x = [[sum(z) for z in zip(sub_v, sub_p)] for sub_v, sub_p in zip(voc_emb,pos_emb)]

        for block in self.blocks:
            x = block(x)

        return list(map(self.out, layernorm(x)))

    def parameters(self):
        emb_pars = [p for p_list in self.voc_emb for p in p_list] + [p for p_list in self.pos_emb for p in p_list]
        block_pars = [p for block in self.blocks for p in block.parameters()]
        out_pars = self.out.parameters()
        return block_pars + out_pars + emb_pars