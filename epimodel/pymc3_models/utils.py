import theano.tensor as T
import numpy as np


def shift_right(t, dist, axis, pad=0.0):
    """
    Return the signal shifted by dist along given axis, padded by `pad`.
    """
    assert dist >= 0
    t = T.as_tensor(t)
    s = list(t.shape.eval())
    axis = axis % len(s)
    dist = min(dist, s[axis])

    lpad = np.full(s[:axis] + [dist] + s[axis + 1 :], pad)

    i2 = [slice(None)] * len(s)
    i2[axis] = slice(None, s[axis] - dist)

    print(lpad, i2)
    res = T.concatenate((lpad, t[i2]), axis=axis)
    assert all(res.shape.eval() == t.shape.eval())
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
