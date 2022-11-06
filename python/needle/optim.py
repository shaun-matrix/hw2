"""Optimization module"""
import needle as ndl
import numpy as np
from numpy.lib import gradient


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            param.data = (1 - self.lr * self.weight_decay) * param.data
            #grad = param.grad.data + param.data * self.weight_decay
            grad = ndl.Tensor(param.grad.realize_cached_data(), dtype=param.dtype)
            if id(param) not in self.u:
                # initial u with key=id(param), value=param.grad
                self.u[id(param)] = (1 - self.momentum) * grad.data
            else:
                self.u[id(param)].data = self.momentum * self.u[id(param)].data \
                  + (1 - self.momentum) * grad.data
            param.data = param.data - self.lr * (self.u[id(param)].data)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            p.data = (1 - self.lr * self.weight_decay) * p.data
            grad = ndl.Tensor(p.grad.realize_cached_data(), dtype=p.dtype)
            if id(p) not in self.m:
                # initial u with key=id(param), value=param.grad
                self.m[id(p)] = (1 - self.beta1) * grad.data
                self.v[id(p)] = (1 - self.beta2) * grad.data * grad.data
            else:
                self.m[id(p)].data = self.beta1 * self.m[id(p)].data \
                  + (1 - self.beta1) * grad.data
                self.v[id(p)].data = self.beta2 * self.v[id(p)].data \
                  + (1 - self.beta2) * grad.data * grad.data
            m_bias_correction = self.m[id(p)].data / (1-pow(self.beta1, self.t))
            v_bias_correction = self.v[id(p)].data / (1-pow(self.beta2, self.t))
            p.data = p.data - self.lr * m_bias_correction.data \
              / (v_bias_correction.data ** 0.5 + self.eps)
        ### END YOUR SOLUTION
