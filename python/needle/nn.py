"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.bias_flag = bias
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features))
        if self.bias_flag:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(self.out_features, 1),\
            (1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias_flag:
            
            mul = X @ self.weight
            return mul + ops.broadcast_to(self.bias, mul.shape)
        else:
            return X @ self.weight
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        batch = shape[0]
        non_batch_dims = 1
        for dim in shape[1:]:
            non_batch_dims *= dim
        return X.reshape((batch, non_batch_dims))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_samples, classes = logits.shape
        z_y = (init.one_hot(classes, y) * logits).sum(axes=(1,))
        return (ops.logsumexp(logits, axes=(1,)) - z_y).sum() / num_samples
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # note: we have to broadcast tensor explicitly in needle
        mini_batch = x.shape[0]
        if self.training == True:
            mean = x.sum(axes=(0,)) / mini_batch
            mean_bc = mean.broadcast_to((mini_batch, self.dim))
            var = ((x - mean_bc) * (x - mean_bc)).sum(axes=(0,)) / mini_batch
            var_bc = var.broadcast_to((mini_batch, self.dim))
            # running_mean and running_var used at evaluation time
            tmp_mean = Tensor(mean.realize_cached_data(), dtype=self.running_mean.dtype)
            tmp_var = Tensor(var.realize_cached_data(), dtype=self.running_var.dtype)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * tmp_mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * tmp_var.data
            return self.weight.broadcast_to((mini_batch, self.dim)) * ((x - mean_bc) \
              / ((var_bc + self.eps) ** 0.5)) + self.bias.broadcast_to((mini_batch, self.dim))
        else:
            return self.weight.broadcast_to((mini_batch, self.dim)) * \
              ((x - self.running_mean.broadcast_to((mini_batch, self.dim))) \
              / ((self.running_var.broadcast_to((mini_batch, self.dim)) + self.eps) ** 0.5)) \
              + self.bias.broadcast_to((mini_batch, self.dim))
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mini_batch = x.shape[0]
        mean = (x.sum(axes=(1,))) / self.dim
        # we broadcast mean and var to compute with x
        mean_bc = (mean.reshape((mini_batch, 1))).broadcast_to((mini_batch, self.dim))
        var = ((x - mean_bc) * (x - mean_bc)).sum(axes=(1,)) / self.dim
        var_bc = (var.reshape((mini_batch, 1))).broadcast_to((mini_batch, self.dim)) \
            + self.eps
        return self.weight.broadcast_to((mini_batch, self.dim)) * (x - mean_bc) * (var_bc ** -0.5) + \
          self.bias.broadcast_to((mini_batch, self.dim))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mask = init.randb(*x.shape, p=1-self.p)
        return x * mask / (1-self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



