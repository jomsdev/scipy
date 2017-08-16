"""Tests for functions in random_projections.py."""

from __future__ import division, print_function, absolute_import

__usage__ = """
Build linalg:
  python setup_linalg.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.linalg.test()'
"""

from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_raises, assert_, assert_allclose)

import pytest

from scipy.linalg import clarckson_woodruff_transform

class TestRandomProjections(object):
    def check_clarkson_woodruff_transform():
        assert_equal([1], [2])
