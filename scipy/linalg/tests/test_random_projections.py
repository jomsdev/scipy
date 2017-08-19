"""Tests for functions in random_projections.py."""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.linalg import clarckson_woodruff_transform

__usage__ = """
Build linalg:
  python setup_linalg.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.linalg.test()'
"""

from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_raises, assert_, assert_allclose)
from nose.tools import (assert_true)


import pytest

# Make some random data with Gaussian distributed values
def make_random_gaussian_matrix(n_rows, n_columns, mu=0, sigma=0.01):
    res = np.random.normal(mu, sigma, n_rows*n_columns)
    return np.reshape(res, (n_rows, n_columns))

class TestRandomProjections(object):

    def test_clarkson_woodruff_transform(self):
        A = make_random_gaussian_matrix(500, 2000)
        assert_true(A.shape == (500, 2000))
