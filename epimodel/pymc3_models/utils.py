import theano.tensor as T
import numpy as np


def shift_right(t, dist, axis, pad=0.0):
    """
    Return the signal shifted by dist along given axis, padded by `pad`.
    """
    assert dist >= 0
    t = T.as_tensor(t)
    if dist == 0:
        return t
    p = T.ones_like(t) * pad

    # Slices
    ts = [slice(None)] * t.ndim
    ts[axis] = slice(None, -dist)  # only for dim > 0

    ps = [slice(None)] * t.ndim
    ps[axis] = slice(None, dist)

    res = T.concatenate((p[ps], t[ts]), axis=axis)
    return res


def convolution(t, weights, axis):
    """
    Computes a linear convolution of tensor by weights.
    
    The result is res[.., i, ..] = w[0] * res[.., i, ..]
    """
    t = T.as_tensor(t)
    res = T.zeros_like(t)
    for i, dp in enumerate(weights):
        res = res + dp * shift_right(t, dist=i, axis=axis, pad=0.0)
    return res


def geom_convolution(t, weights, axis):
    """
    Computes a linear convolution of log(tensor) by weights, returning exp(conv_res).
    
    Can be also seen as geometrical convolution.
    The result is res[.., i, ..] = w[0] * res[.., i, ..]
    """
    t = T.as_tensor(t)
    res = T.ones_like(t)
    for i, dp in enumerate(weights):
        res = res * shift_right(t, dist=i, axis=axis, pad=1.0) ** dp
    return res
