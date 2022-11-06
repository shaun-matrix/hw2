"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #return multiply(out_grad, power_scalar(node.inputs[0], self.scalar))
        #grad = mul_scalar(power_scalar(node.inputs[0], self.scalar-1), self.scalar)
        #return multiply(out_grad, grad)
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar-1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        l_partial_adjoint = divide(out_grad, rhs)
        r_partial_adjoint = negate(divide(multiply(out_grad, lhs), power_scalar(rhs, 2)))
        return l_partial_adjoint, r_partial_adjoint
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, -1, -2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = list(node.inputs[0].shape)
        out_shape = list(self.shape)
        for _ in range(len(out_shape) - len(in_shape)):
            in_shape.insert(0, 0)
        # partial_gradient_adjoint may be re-summated.
        partial_gradient_adjoint = out_grad
        summation_cnt = 0
        for axe in range(len(out_shape)):
            if in_shape[axe] != out_shape[axe]:
                partial_gradient_adjoint = summation(partial_gradient_adjoint, 
                axes=axe-summation_cnt)
                # dimensions of partial_gradient_adjoint will decrease if summation is operated
                summation_cnt += 1
        return reshape(partial_gradient_adjoint, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = list(node.shape) if isinstance(node.shape, tuple) \
          else [node.shape]
        partial_gradient_adjoint = out_grad

        # default self.axes=None
        axes = self.axes if self.axes is not None else tuple(range(len(input_shape)))
        for axe in axes:
            # broadcast tensor directly
            if axe == 0:
                output_shape.insert(axe, input_shape[axe])
                partial_gradient_adjoint = broadcast_to(partial_gradient_adjoint\
                  , tuple(output_shape))
            # first reshape tensor then broadcast
            else:
                output_shape.append(1)
                reshape_shape = tuple(output_shape)
                output_shape[-1] = input_shape[axe]
                broadcast_shape = tuple(output_shape)
                partial_gradient_adjoint = broadcast_to(reshape(\
                  partial_gradient_adjoint, reshape_shape), broadcast_shape)
        return partial_gradient_adjoint
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lshape_len = len(lhs.shape)
        rshape_len = len(rhs.shape)
        if (lshape_len>2 or rshape_len>2) and (lshape_len != rshape_len):
            extra_len = abs(lshape_len - rshape_len)
            if lshape_len > rshape_len:
                return matmul(out_grad, transpose(rhs)), \
                  summation(matmul(transpose(lhs), out_grad), axes=tuple(range(extra_len)))
            else:
                return summation(matmul(out_grad, transpose(rhs)), axes=tuple(range(extra_len))), \
                  matmul(transpose(lhs), out_grad)
        else:
            return matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # not using this in-place, because this will cause problem in gradient check
        #a[a<0] = 0
        return array_api.where(a>0, a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        out_grad_array = out_grad.numpy()
        partial_adjoint = array_api.where(out>0, out_grad_array, 0)
        return Tensor(partial_adjoint)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return array_api.log(array_api.sum(array_api.exp(Z - \
          array_api.amax(Z, axis=self.axes, keepdims=True)), axis=self.axes)) \
          + array_api.amax(Z, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # convert from Tensor to numpy
        input_realized = node.inputs[0].realize_cached_data()
        input_shape = input_realized.shape
        # convert to list because this shape will be broadcasted.
        cur_shape = list(node.shape)
        out_grad_bc = out_grad.realize_cached_data()

        # default self.axes=None
        axes = self.axes if self.axes is not None else tuple(range(len(input_shape)))
        # first broadcast out_grad
        for axe in axes:
            cur_shape.insert(axe, 1)
        out_grad_bc = array_api.reshape(out_grad_bc, tuple(cur_shape))
        out_grad_bc = array_api.broadcast_to(out_grad_bc, input_shape)
        '''
        for axe in axes:
            if axe == 0:
                cur_shape.insert(0, input_shape[0])
                out_grad_bc = array_api.broadcast_to(out_grad_bc, tuple(cur_shape))
            else:
                cur_shape.append(1)
                out_grad_bc = array_api.reshape(out_grad_bc, tuple(cur_shape))
                cur_shape[-1] = input_shape[axe]
                out_grad_bc = array_api.broadcast_to(out_grad_bc, tuple(cur_shape))
        '''
        # calculate softmax(Z-max(Z))
        max_of_input = array_api.amax(input_realized, axis=self.axes, keepdims=True)
        softmax_grad = array_api.exp(input_realized - max_of_input) / \
          array_api.sum(array_api.exp(input_realized - max_of_input), \
          axis=self.axes, keepdims=True)

        return Tensor(out_grad_bc * softmax_grad)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
