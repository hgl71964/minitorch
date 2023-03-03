import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def _check_shape_larger(out_shape, in_shape) -> bool:
    n1 = len(out_shape)
    n2 = len(in_shape)

    if n1 == n2:
        for k in range(n1):
            i = out_shape[k]
            j = in_shape[k]
            if i < j:
                return False
    elif n1 > n2:
        diff = n1 - n2
        new_in_shape = tuple([1 for i in range(diff)]) + tuple(in_shape)
        for k in range(n1):
            i = out_shape[k]
            j = new_in_shape[k]
            if i < j:
                return False
    else:
        return False

    return True


def tensor_map(fn):
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 2.2.
        # raise NotImplementedError("Need to implement for Task 2.2")
        # NOTE: the idea is write to tensor storage
        assert (_check_shape_larger(out_shape,
                                    in_shape)), f"{out_shape} - {in_shape}"

        # total output size
        out_size = 1
        for i in out_shape:
            out_size *= i

        for out_ordinal in range(out_size):
            # make index buffer
            big_index = np.array(out_shape)
            in_index = np.array(in_shape)
            # compute index
            to_index(out_ordinal, out_shape, big_index)
            broadcast_index(big_index, out_shape, in_shape, in_index)
            # print(big_index)
            # print(in_index)
            # write
            in_strorage_val = in_storage[index_to_position(
                in_index, in_strides)]
            out[out_ordinal] = fn(in_strorage_val)

    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 2.2.
        # raise NotImplementedError("Need to implement for Task 2.2")
        assert (_check_shape_larger(out_shape,
                                    a_shape)), f"{out_shape} - {a_shape}"
        assert (_check_shape_larger(out_shape,
                                    b_shape)), f"{out_shape} - {b_shape}"

        # total output size
        out_size = 1
        for i in out_shape:
            out_size *= i

        for out_ordinal in range(out_size):
            # make index buffer
            a_index = np.array(a_shape)
            b_index = np.array(b_shape)
            out_index = np.array(out_shape)
            # compute index
            to_index(out_ordinal, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # write
            a_strorage_val = a_storage[index_to_position(a_index, a_strides)]
            b_strorage_val = b_storage[index_to_position(b_index, b_strides)]
            out[out_ordinal] = fn(a_strorage_val, b_strorage_val)

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """
    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides,
                reduce_dim):
        # TODO: Implement for Task 2.2.
        # raise NotImplementedError("Need to implement for Task 2.2")

        # total output size
        a_size = 1
        for i in a_shape:
            a_size *= i

        for a_ordinal in range(a_size):
            # make index buffer
            a_index = np.array(a_shape)

            # compute index
            to_index(a_ordinal, a_shape, a_index)

            out_correspond_index = [0] * len(a_index)
            for i, v in enumerate(a_index):
                if i != reduce_dim:
                    out_correspond_index[i] = v
                else:
                    out_correspond_index[i] = 0

            out_position = index_to_position(out_correspond_index, out_strides)

            # write
            a_strorage_val = a_storage[a_ordinal]
            out[out_position] = fn(out[out_position], a_strorage_val)

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`TensorData` : new tensor
    """
    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
