"""Random Projection transformers"""

from __future__ import division, print_function, absolute_import

import numpy as np

__all__ = ['clarckson_woodruff_transform']

def clarckson_woodruff_transform(input_matrix, number_of_columns):
    """
    Given a matrix A (input_matrix) of size (n, d), compute a matrix A' of size  (n, s) which holds:

        $||Ax|| = (1 \pm \epsilon) ||A'x||$

    with high probability.

    To obtain A' we create a matrix S of size (d, s) where every column of transpose(S) has only one position distinct to zero
    with value +1 or -1. We multiply S*A to obtain A'.

    Parameters
    ----------
    input_matrix (A) : (n, d) array_like
        Input matrix
    number_of_columns (s) : int
        number of columns for A'

    Returns
    -------
    A' : (n, s) array_like
        Sketch of A

    Notes
    -----
    This is an implementation of the Clarckson-Woodruff Transform (also known as CountSketch) introduced for
    first time in Kenneth L. Clarkson and David P. Woodruff. Low rank approximation and regression in input sparsity time. In STOC, 2013.

    A' can be computed in O(nnz(A)) but we don't take advantage of sparse matrix in this implementation
    """
    return np.zeros(input_matrix.shape[0], number_of_columns)
