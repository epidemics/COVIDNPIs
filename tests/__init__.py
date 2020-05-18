import unittest
import numpy as np
import numpy.testing as npt


class NumpyTestCase(unittest.TestCase):
    """
    Test class that includes all of the non-redundant numpy test assertions
    https://docs.scipy.org/doc/numpy/reference/routines.testing.html#asserts
    """

    def assert_almost_equal(self, *args, **kwargs):
        npt.assert_almost_equal(*args, **kwargs)

    def assert_approx_equal(self, *args, **kwargs):
        npt.assert_approx_equal(*args, **kwargs)

    def assert_array_almost_equal(self, *args, **kwargs):
        npt.assert_array_almost_equal(*args, **kwargs)

    def assert_allclose(self, *args, **kwargs):
        npt.assert_allclose(*args, **kwargs)

    def assert_array_almost_equal_nulp(self, *args, **kwargs):
        npt.assert_array_almost_equal_nulp(*args, **kwargs)

    def assert_array_max_ulp(self, *args, **kwargs):
        npt.assert_array_max_ulp(*args, **kwargs)

    def assert_array_equal(self, *args, **kwargs):
        npt.assert_array_equal(*args, **kwargs)

    def assert_array_less(self, *args, **kwargs):
        npt.assert_array_less(*args, **kwargs)

    def assert_string_equal(self, *args, **kwargs):
        npt.assert_string_equal(*args, **kwargs)


class PandasTestCase(NumpyTestCase):
    def assert_dtype(self, series, dtype):
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)

        if series.name is not None:
            name = f"{series.name!r} series"
        else:
            name = "series"

        self.assertTrue(
            np.issubdtype(series.dtype, dtype),
            f"{name} dtype {series.dtype.name!r} != {dtype.name!r}",
        )
