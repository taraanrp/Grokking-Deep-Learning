"""Implementasi pedagogis NumPy berdasarkan operasi Bab 13 (Trask, Manning).

Penyesuaian struktur: traversal topologis menggantikan children-count rekursif.
Gradien adalah ndarray, loss mean konsisten, indeks berulang memakai scatter-add.
Tidak mendukung higher-order gradients atau seluruh broadcasting NumPy matmul.
Kode ini juga ditampilkan lengkap dan bertahap di notebook Bab 13.
"""
import numpy as np


def unbroadcast(gradient, shape):
    """Undo broadcasting: setiap penggunaan elemen menjumlahkan kontribusi gradien."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, autograd=False, parents=(), op='', backward=None):
        self.data = np.array(data, dtype=float)
        self.autograd = autograd
        self.grad = np.zeros_like(self.data)
        self.parents = tuple(parents)
        self.op = op
        self._backward = backward or (lambda g: None)

    def __repr__(self):
        return f'Tensor({self.data!r}, op={self.op!r})'

    @staticmethod
    def wrap(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    def _result(self, data, parents, op, backward):
        return Tensor(data, any(p.autograd for p in parents), parents, op, backward)

    def _accumulate(self, gradient):
        if self.autograd:
            self.grad += unbroadcast(np.asarray(gradient), self.data.shape)

    def __add__(self, other):
        other = Tensor.wrap(other)
        def backward(g):
            self._accumulate(g)
            other._accumulate(g)
        return self._result(self.data+other.data,(self,other),'add',backward)

    __radd__ = __add__

    def __neg__(self):
        return self._result(-self.data,(self,),'neg',lambda g:self._accumulate(-g))

    def __sub__(self, other):
        return self + (-Tensor.wrap(other))

    def __mul__(self, other):
        other = Tensor.wrap(other)
        def backward(g):
            self._accumulate(g*other.data)
            other._accumulate(g*self.data)
        return self._result(self.data*other.data,(self,other),'mul',backward)

    __rmul__ = __mul__

    def mm(self, other):
        other = Tensor.wrap(other)
        assert self.data.ndim == other.data.ndim == 2
        def backward(g):
            self._accumulate(g@other.data.T)
            other._accumulate(self.data.T@g)
        return self._result(self.data@other.data,(self,other),'mm',backward)

    __matmul__ = mm

    def transpose(self):
        assert self.data.ndim == 2
        return self._result(self.data.T,(self,),'transpose',lambda g:self._accumulate(g.T))

    def sum(self, dim=None):
        def backward(g):
            restored = np.expand_dims(g, dim) if dim is not None else g
            self._accumulate(np.ones_like(self.data)*restored)
        return self._result(self.data.sum(axis=dim),(self,),'sum',backward)

    def mean(self, dim=None):
        count = self.data.size if dim is None else self.data.shape[dim]
        return self.sum(dim)*(1/count)

    def expand(self, dim, copies):
        expanded = np.repeat(np.expand_dims(self.data,dim),copies,axis=dim)
        return self._result(expanded,(self,),'expand',lambda g:self._accumulate(g.sum(axis=dim)))

    def sigmoid(self):
        value = np.exp(-np.logaddexp(0,-self.data))
        return self._result(value,(self,),'sigmoid',lambda g:self._accumulate(g*value*(1-value)))

    def tanh(self):
        value = np.tanh(self.data)
        return self._result(value,(self,),'tanh',lambda g:self._accumulate(g*(1-value**2)))

    def index_select(self, indices):
        indices = indices.data if isinstance(indices,Tensor) else np.asarray(indices)
        assert np.all(indices == indices.astype(int))
        indices = indices.astype(int)
        def backward(g):
            if self.autograd:
                np.add.at(self.grad,indices,g)
        return self._result(self.data[indices],(self,),'index_select',backward)

    def cross_entropy(self, target):
        target = target.data if isinstance(target,Tensor) else np.asarray(target)
        target = target.astype(int).ravel()
        assert self.data.ndim == 2 and len(target) == len(self.data)
        z = self.data-self.data.max(axis=1,keepdims=True)
        log_probs = z-np.log(np.exp(z).sum(axis=1,keepdims=True))
        prob = np.exp(log_probs)
        value = -log_probs[np.arange(len(target)),target].mean()
        def backward(g):
            delta = prob.copy()
            delta[np.arange(len(target)),target] -= 1
            self._accumulate(g*delta/len(target))
        return self._result(value,(self,),'cross_entropy',backward)

    def detach(self):
        return Tensor(self.data.copy(), self.autograd)

    def backward(self, gradient=None):
        # Setiap node diproses setelah semua anak dalam graph aktif selesai.
        order=[]; visited=set()
        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            order.append(node)
        visit(self)
        # Satu panggilan backward = satu graph; jangan membawa gradien batch lama.
        for node in order:
            node.grad = np.zeros_like(node.data)
        if gradient is None:
            gradient=np.ones_like(self.data)
        if isinstance(gradient,Tensor):
            gradient=gradient.data
        self.grad=np.asarray(gradient,dtype=float)
        assert self.grad.shape == self.data.shape
        for node in reversed(order):
            if node.autograd:
                node._backward(node.grad)
        for node in order:
            assert np.isfinite(node.grad).all(), f'Gradien tidak finite: {node.op}'


class SGD:
    def __init__(self, parameters, alpha=.1):
        self.parameters=list(parameters)
        self.alpha=alpha

    def zero(self):
        for p in self.parameters:
            p.grad.fill(0)

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= self.alpha*p.grad
            assert np.isfinite(p.data).all()
        if zero:
            self.zero()


class Layer:
    def get_parameters(self):
        return []

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)


class Linear(Layer):
    def __init__(self,n_inputs,n_outputs,bias=True):
        self.weight=Tensor(np.random.randn(n_inputs,n_outputs)*np.sqrt(2/n_inputs),True)
        self.bias=Tensor(np.zeros(n_outputs),True) if bias else None

    def forward(self,x):
        out=x.mm(self.weight)
        return out+self.bias.expand(0,len(x.data)) if self.bias is not None else out

    def get_parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])


class Sequential(Layer):
    def __init__(self,layers=None):
        self.layers=list(layers or [])

    def add(self,layer):
        self.layers.append(layer)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

    def get_parameters(self):
        return [p for layer in self.layers for p in layer.get_parameters()]


class MSELoss(Layer):
    def forward(self,pred,target):
        difference=pred-target
        return (difference*difference).mean()


class Tanh(Layer):
    def forward(self,x):
        return x.tanh()


class Sigmoid(Layer):
    def forward(self,x):
        return x.sigmoid()


class Embedding(Layer):
    def __init__(self,vocab_size,dim):
        self.weight=Tensor((np.random.rand(vocab_size,dim)-.5)/dim,True)

    def forward(self,indices):
        return self.weight.index_select(indices)

    def get_parameters(self):
        return [self.weight]


class CrossEntropyLoss(Layer):
    def forward(self,pred,target):
        return pred.cross_entropy(target)


class RNNCell(Layer):
    def __init__(self,n_inputs,n_hidden,n_output,activation='sigmoid'):
        self.n_hidden=n_hidden
        self.w_ih=Linear(n_inputs,n_hidden)
        self.w_hh=Linear(n_hidden,n_hidden)
        self.w_ho=Linear(n_hidden,n_output)
        self.activation={'sigmoid':Sigmoid,'tanh':Tanh}[activation]()

    def forward(self,x,hidden):
        hidden=self.activation(self.w_ih(x)+self.w_hh(hidden))
        return self.w_ho(hidden),hidden

    def init_hidden(self,batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)),True)

    def get_parameters(self):
        return self.w_ih.get_parameters()+self.w_hh.get_parameters()+self.w_ho.get_parameters()
