#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" MYSPH : SPH simulation for odorant plume simulation
    Copyright (C) 2019  Batuhan Başerdem

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as n
import scipy.sparse as s

# Error handling
class Error(Exception):
    """ Base class for errors """

class TenMatInvalidInputError(Error):
    """ Raise this error when class encounters dimensional mismatch """

class TenMatMismatchDimError(Error):
    """ Raise this error when class encounters dimensional mismatch """

class TenMatNonSparseError(Error):
    """ Raise this error when class deals with non-sparse values """

# Class to create vector matrices
class TenMat():
    """
    Class to store a tensor of sparse arrays.
        This class enables the simultaneous use of indices,
        while doing algebra on only one. The two types of indices are;
        -tensor:    The index where all algebra is done. Assumption is that
                    this is going to be a 2D or 3D tensor.
        -sparse:    The index which is to be sparse. Is in the form of a sparse
                    matrix (or a full vector if it is a vector)
    """
    def __init__(self,
                 sparsity=(n.array([]), n.array([])),
                 values=n.empty((0, 1)),
                 tensor_shape=None,
                 sparse_shape=None):
        """
        Constructor for TenMat
        Take numpy array of sparse matrices, and merge them in one.
            -sparsity:  A tuple of length 2, which has sparsity indices on the
                        slots.
                        * This is NOT NEEDED if data is a numpy vector, or only
                        has one column
            -values:    Matrix values at sparse indices.
                        Must be provided as a numpy array;
                            - First dimension is the sparsity position
                            - Any other dimensions are the dimensions among
                                tensor index
            -tensor_shape:  Shape of the tensor index. If it exists, reshapes
                            the values matrix to fit this data.
            -sparse_shape:  Shape of the sparse index. Either this or sparsity
                            must be provided.
        """

        # Write sparse shape
        if sparse_shape is None:
            if len(values.shape) > 2:
                raise TenMatInvalidInputError
            self.sp_dim = values.shape[:]
        else:


        self.sparseness = args
        self.dim = len(args)
        self.shape = self.data[0].shape
        self.dtype = self.data[0].dtype
        self.nnz = max(a.nnz for a in self.data)
        # Checks
        for i in range(self.dim):
            # Shape check
            if self.data[i].shape != self.shape:
                raise VecMatInvalidInputError
            # Type check
            elif self.data[i].dtype != self.dtype:
                raise VecMatInvalidInputError
            # Sparseness check
            elif not isinstance(self.data[i], s.spmatrix):
                raise VecMatInvalidInputError

    def __eq__(self, mat):
        """ Test equality """
        if isinstance(mat, VecMat):
            if (self.dim == mat.dim) and (self.shape == mat.shape):
                for i in range(self.dim):
                    if (self.data[i] != mat.data[i]).nnz:
                        return False
                return True
        if isinstance(mat, (n.ndarray, s.spmatrix)):
            if self.shape == mat.shape:
                for i in range(self.dim):
                    if self.data[i] != mat:
                        return False
            return True
        return False

    def __mul__(self, mat):
        """ Overloading of * operator for these matrices """
        # SELF
        if isinstance(mat, VecMat):
            self.hadamart_self(mat)
        # NUMPY ARRAY or SPARSEARRAY MULTIPLY
        if isinstance(mat, (n.ndarray, s.spmatrix)):
            # IF IS VECTOR
            if len(mat.shape) == 1:
                return self.hadamart_vector(mat)
            # IF IS MATRIX
            if len(mat.shape) == 2:
                return self.hadamart_matrix(mat)
        num = (int, float, complex, n.int, n.float, n.double, n.complex)
        if isinstance(mat, num):
            out = [a * mat for a in self.data]
            return VecMat(*out)
        # Default
        return NotImplemented

    def __rmul__(self, mat):
        """ Reverse operation of elementwise product """
        return self * mat

    def __matmul__(self, mat):
        """ Overloading of @ operator for these matrices """
        # SELF PRODUCT
        if isinstance(mat, VecMat):
            self.multiply_self(mat)
        # NUMPY ARRAY or SPARSEARRAY MULTIPLY
        if isinstance(mat, (n.ndarray, s.spmatrix)):
            # IF IS VECTOR
            if len(mat.shape) == 1:
                return self.multiply_vector(mat)
            # IF IS MATRIX
            if len(mat.shape) == 2:
                return self.multiply_matrix(mat)
        num = (int, float, complex, n.int, n.float, n.double, n.complex)
        if isinstance(mat, num):
            out = [a @ mat for a in self.data]
            return VecMat(*out)
        # Default
        return NotImplemented

    def __rmatmul__(self, mat):
        """ Reverse operation of matrix multiplication, abuse transposes """
        out = self.transpose() @ mat.transpose()
        return out.transpose()

    def multiply_self(self, mat):
        """ Calculates matrix product with another instance """
        if (self.dim == mat.dim) and (self.shape[1] == mat.shape[0]):
            out = [a.dot(b) for a, b in zip(self.data, mat.data)]
            return VecMat(*out)
        raise VecMatMismatchDimError

    def multiply_matrix(self, mat):
        """ Calculates matrix product with another matrix """
        # PARTICLE-WISE
        if self.shape[1] == mat.shape[0]:
            out = [s.csc_matrix(a.dot(mat)) for a in self.data]
            return VecMat(*out)
        # SPACE-WISE
        if self.dim == mat.shape[0]:
            out = [s.csc_matrix(
                sum(a*b for a, b in zip(self.data, mat[:, i])),
                dtype=self.dtype) for i in range(mat.shape[1])]
            return VecMat(*out)
        raise VecMatMismatchDimError

    def multiply_vector(self, mat):
        """ Calculates matrix product with a vector """
        # PARTICLE-WISE (Auto-contract to second index)
        if self.shape[1] == mat.shape[0]:
            out = n.column_stack([a.dot(mat) for a in self.data])
            return out.transpose()
        # SPACE-WISE
        if self.dim == mat.shape[0]:
            out = s.csc_matrix(
                sum(a*b for a, b in zip(self.data, mat)),
                dtype=self.dtype)
            return out
        raise VecMatMismatchDimError

    def hadamart_self(self, mat):
        """ Calculates elementwise product with another instance """
        if (self.dim == mat.dim) and (self.shape == mat.shape):
            out = [a.multiply(b) for a, b in zip(self.data, mat.data)]
            return VecMat(*out)
        raise VecMatMismatchDimError

    def hadamart_matrix(self, mat):
        """ Calculates elementwise product with another matrix """
        # PARTICLE-WISE
        if self.shape == mat.shape:
            out = [s.csc_matrix(a.multiply(mat)) for a in self.data]
            return VecMat(*out)
        raise VecMatMismatchDimError

    def hadamart_vector(self, mat):
        """ Calculates elementwise product with a vector """
        # PARTICLE-WISE
        if self.shape[1] == mat.shape[0]:
            out = [a.multiply(mat) for a in self.data]
            return VecMat(*out)
        # SPACE-WISE
        if self.dim == mat.shape[0]:
            out = [a*b for a, b in zip(self.data, mat)]
            return VecMat(*out)
        raise VecMatMismatchDimError

    def dot(self, mat):
        """ Takes a (spatial) dot product between two class instances """
        return sum(a.multiply(b) for a, b in zip(self.data, mat.data))

    def contract(self, caxis=None):
        """ Contract a particle index, giving a concatanated array """
        if caxis is None:
            return self
        return n.column_stack([a.sum(axis=caxis) for a in self.data])

    def transpose(self):
        """ Transpose the index elements """
        out = [a.transpose() for a in self.data]
        return VecMat(*out)

def randomVecMat(shape, dimension, density=0.01):
    """ Generate a random instance of VecMat """
    # Generate indices
    sto = s.random(shape[0], shape[1], density=density, format='csc')
    nonzero = sto.nnz
    indices = sto.nonzero()
    # Generate list of these
    values = [n.random.rand(nonzero) for _ in range(dimension)]
    holder = [s.csc_matrix((val, indices), shape=shape) for val in values]
    return VecMat(*holder)


if __name__ == "__main__":
    # Do a display of initialization if called explicitly
    pass
