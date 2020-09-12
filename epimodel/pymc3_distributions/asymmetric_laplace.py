"""
:code:`asymmetric_laplace.py`

Asymmetric Laplace Distribution, with location parameter 0. This is used as our NPI Effectiveness prior.

See also: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
"""
import pymc3.distributions.continuous as continuous
import theano.tensor as tt
import numpy as np


class AsymmetricLaplace(continuous.Continuous):
    """
    Assymetric Laplace Distribution

    See also: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
    """

    def __init__(self, scale, symmetry, testval=0.0, *args, **kwargs):
        """
        Constructor

        :param scale: scale parameter
        :param symmetry: asymmetry parameter. Reduces to a normal laplace distribution with value 1
        """
        self.scale = tt.as_tensor_variable(scale)
        self.symmetry = tt.as_tensor_variable(symmetry)

        super().__init__(*args, **kwargs, testval=testval)

    def random(self, point=None, size=None):
        """
        Draw random samples from this distribution, using the inverse CDF method.

        :param point: not used
        :param size: size of sample to draw
        :return: Samples
        """
        if point is not None:
            raise NotImplementedError('Random not implemented with point specified')

        if size is not None:
            u = np.random.uniform(size=size)
            x = - tt.log((1 - u) * (1 + self.symmetry ** 2)) / (self.symmetry * self.scale) * (
                    u > ((self.symmetry ** 2) / (1 + self.symmetry ** 2))) + self.symmetry * tt.log(
                u * (1 + self.symmetry ** 2) / (self.symmetry ** 2)) / self.scale * (
                        u < ((self.symmetry ** 2) / (1 + self.symmetry ** 2)))

            return x

        u = np.random.uniform()
        if u > (self.symmetry ** 2) / (1 + self.symmetry ** 2):
            x = - tt.log((1 - u) * (1 + self.symmetry ** 2)) / (self.symmetry * self.scale)
        else:
            x = self.symmetry * tt.log(u * (1 + self.symmetry ** 2) / (self.symmetry ** 2)) / self.scale

        return x

    def logp(self, value):
        """
        Compute logp.

        :param value: evaluation point
        :return: log probability at evaluation point
        """
        return tt.log(self.scale / (self.symmetry + (self.symmetry ** -1))) + (
                -value * self.scale * tt.sgn(value) * (self.symmetry ** tt.sgn(value)))
