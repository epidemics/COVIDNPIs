import pytest
from pytest import approx
import numpy as np

theano = pytest.importorskip("theano")
pm = pytest.importorskip("pymc3")

from epimodel.pymc3_models import utils
import theano.tensor as T

A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],])
W = [1, 2, 1]


def test_shift():
    assert utils.shift_right(A, 2, axis=1).eval() == approx(
        np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10]])
    )
    assert utils.shift_right(A, 1, axis=-1).eval() == approx(
        np.array([[0, 1, 2, 3], [0, 5, 6, 7], [0, 9, 10, 11]])
    )
    assert utils.shift_right(A, 1, axis=0).eval() == approx(
        np.array([[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]])
    )

    assert utils.shift_right(A, 10, axis=0).eval() == approx(np.zeros_like(A))
    assert utils.shift_right(A, 10, axis=1).eval() == approx(np.zeros_like(A))

    a = np.random.normal(size=(1, 2, 2, 1, 4))
    assert utils.shift_right(a, 2, axis=1).eval() == approx(np.zeros_like(a))
    assert utils.shift_right(a, 3, axis=-1).eval() != approx(np.zeros_like(a))
    assert utils.shift_right(a, 1, axis=-2).eval() == approx(np.zeros_like(a))


def test_convolution():
    for axis in [0, 1]:
        assert utils.convolution(A, [0, 0, 0, 0, 0], axis).eval() == approx(0)
        assert utils.convolution(A, [], axis).eval() == approx(0)
        assert utils.convolution(A, [1], axis).eval() == approx(A)

    assert utils.convolution(A, W, 1).eval() == approx(
        np.array([[1, 4, 8, 12], [5, 16, 24, 28], [9, 28, 40, 44]])
    )
    assert utils.convolution(A, W, 0).eval() == approx(
        np.array([[1, 2, 3, 4], [7, 10, 13, 16], [20, 24, 28, 32]])
    )


def test_geom_convolution():
    W2 = [0.1, 0.3, 0.5, 0.1]
    for axis in [0, 1]:
        assert utils.geom_convolution(A, [0, 0, 0, 0, 0], axis).eval() == approx(1)
        assert utils.geom_convolution(A, [], axis).eval() == approx(1)
        assert utils.geom_convolution(A, [1], axis).eval() == approx(A)
        assert utils.geom_convolution(A, W, axis).eval() == approx(
            T.exp(utils.convolution(T.log(A), W, axis)).eval()
        )
        assert utils.geom_convolution(A, W2, axis).eval() == approx(
            T.exp(utils.convolution(T.log(A), W2, axis)).eval()
        )
