#!/usr/bin/python
"""
    Test methods for the math module of SPH
"""

# Builtin modules
import unittest as u
import copy

# Third party modules
import numpy as n
import scipy.sparse as s

# Self modules
import odorseeker.sph.math as m

D = 2
N = 100
M = 200
R = .05

class SphMathVecMatTestCase(u.TestCase):
    """
        Test cases for generation of VecMat class
    """
    def setUp(self):
        """ Set up random example instances to test functions on. """
        # Create random instance of VecMat class
        self.vecmat = m.randomVecMat((M, N), D, density=R)
        self.vecmat_sparse = self.vecmat.data

    def test_shape(self):
        """ Test if shape is all well set """
        self.assertEqual(self.vecmat.shape, (M, N))

    def test_transpose(self):
        """ Test if transpose works """
        self.assertEqual(self.vecmat.transpose().transpose(), self.vecmat)

    def test_equality(self):
        """ Test if equality works """
        sto = copy.deepcopy(self.vecmat)
        self.assertEqual(self.vecmat, sto)

    def test_inequality(self):
        """ Test if inequality works """
        sto = m.randomVecMat((M, N), D, density=R)
        self.assertFalse(self.vecmat == sto)

    def test_dimension(self):
        """ Test if the dimension is set properly """
        self.assertEqual(self.vecmat.dim, D)

    def test_nnz(self):
        """ Test if nnz is accurately determined """
        max_nonzero = max(a.nnz for a in self.vecmat.data)
        self.assertEqual(self.vecmat.nnz, max_nonzero)

if __name__ == '__main__':
    u.main()
