"""
FEBasis.py
Author: Benjamin Floyd

Contains the finite local and reference basis functions.
"""
from collections import namedtuple

import numpy as np


class IntervalBasis1D(object):
    """

    Parameters
    ----------
    basis_type
    """
    def __init__(self, basis_type):
        if isinstance(basis_type, int):
            self._basis_type = _BASIS_ALIAS.get(basis_type, None)
        elif isinstance(basis_type, str):
            bstr = basis_type.lower()
            self._basis_type = _BASIS_ALIAS.get(bstr, None)
        else:
            raise TypeError('Basis Type must be a string identifier or an integer code.')

        # This will be overwritten by the local basis function
        self._basis_idx = None
        self._derivative_order_xhat = None
        self._xhat = None

    @property
    def basis_type(self):
        return self._basis_type

    def __fe_reference_basis(self, derivative_order):
        # Set the derivative order
        self._derivative_order_xhat = derivative_order

        if self._basis_type == 101:  # linear basis
            # Set up cases. Dictionary keys are defined as (basis_idx, derivative_order_xhat)
            basis_dispatcher = {
                (0, 0): lambda x_hat: 1 - x_hat,  # Psi_hat_1
                (0, 1): lambda x_hat: -1,  # diff(Psi_hat_1, x_hat)

                (1, 0): lambda x_hat: x_hat,  # Psi_hat_2
                (1, 1): lambda x_hat: 1  # diff(Psi_hat_2, x_hat)
            }

            ref_basis = basis_dispatcher.get((self._basis_idx, self._derivative_order_xhat))
            try:
                return ref_basis(self._xhat)
            except TypeError:
                if self._derivative_order_xhat > 1:
                    return 0.0
                elif self._basis_idx not in [0, 1]:
                    raise KeyError('Basis index must be in [0, 1]. Given: {}'.format(self._basis_idx))
                else:
                    raise KeyError('Derivative order must be positive.')

        elif self._basis_type == 102:  # quadratic basis
            # Set up cases. Dictionary keys are defines as (basis_idx, derivative_order)
            basis_dispatcher = {
                (0, 0): lambda x_hat: 2 * x_hat**2 - 3 * x_hat + 1,  # Psi_hat_0
                (0, 1): lambda x_hat: 4 * x_hat - 3,  # diff(Psi_hat_0, x_hat)
                (0, 2): lambda x_hat: 4,  # diff(Psi_hat_0, x_hat$2)

                (1, 0): lambda x_hat: 2 * x_hat**2 - x_hat,  # Psi_hat_1
                (1, 1): lambda x_hat: 4 * x_hat - 1,  # diff(Psi_hat_1, x_hat)
                (1, 2): lambda x_hat: 4,  # diff(Psi_hat_1, x_hat$2)

                (2, 0): lambda x_hat: -4 * x_hat**2 + 4 * x_hat,  # Psi_hat_2
                (2, 1): lambda x_hat: -8 * x_hat + 4,  # diff(Psi_hat_2, x_hat)
                (2, 2): lambda x_hat: -8  # diff(Psi_hat_2, x_hat$2)
            }

            ref_basis = basis_dispatcher.get((self._basis_idx, self._derivative_order_xhat))
            try:
                return ref_basis(self._xhat)
            except TypeError:
                if self._derivative_order_xhat > 2:
                    return 0.0
                elif self._basis_idx not in [0, 1]:
                    raise KeyError('basis index must be in [0,1,2]. Given: {}'.format(self._basis_idx))
                else:
                    raise KeyError('derivative order must be positive.')

    def fe_local_basis(self, x, vertices, basis_idx, derivative_order):
        """

        Parameters
        ----------
        x
        vertices
        basis_idx
        derivative_order

        Returns
        -------

        """
        # Set the basis index
        self._basis_idx = basis_idx

        # The vertices of the nth finite element En
        x_n, x_np1 = vertices

        # Step size of the element
        h = x_np1 - x_n

        # The affine mapping from the local interval [x_n, x_(n+1)] to the reference interval [0, 1]
        self._xhat = (x - x_n) / h

        return self.__fe_reference_basis(derivative_order) * (1 / h)**derivative_order


class TriangularBasis2D(IntervalBasis1D):
    """

    Parameters
    ----------
    basis_type
    """

    def __init__(self, basis_type):
        # Make a quick adjustment to the basis type string identifiers to make them unique
        if isinstance(basis_type, str) and not basis_type.endswith('2D_tri'):
            basis_type += '2D_tri'
        super().__init__(basis_type)

        # These will be overwritten by the local basis function
        self._derivative_order_yhat = None
        self._yhat = None

    def __fe_reference_basis(self, derivative_order):
        # Set deriviative orders
        self._derivative_order_xhat, self._derivative_order_yhat = derivative_order
        if self._basis_type == 201:  # linear basis
            # Set up casis. Dictionary keys are defined as (basis_idx, derivative_order_x, derivative_order_y)
            basis_dispatcher = {
                (0, 0, 0): lambda x_hat, y_hat: -x_hat - y_hat + 1,  # Psi_hat_0
                (0, 1, 0): lambda x_hat, y_hat: -1,  # diff(Psi_hat_0, x_hat)
                (0, 0, 1): lambda x_hat, y_hat: -1,  # diff(Psi_hat_0, y_hat)

                (1, 0, 0): lambda x_hat, y_hat: x_hat,  # Psi_hat_1
                (1, 1, 0): lambda x_hat, y_hat: 1,  # diff(Psi_hat_1, x_hat)

                (2, 0, 0): lambda x_hat, y_hat: y_hat,  # Psi_hat_2
                (2, 0, 1): lambda x_hat, y_hat: 1  # diff(Psi_hat_2, y_hat)
            }

            ref_basis = basis_dispatcher.get((self._basis_idx,
                                              self._derivative_order_xhat,
                                              self._derivative_order_yhat))
            try:
                return ref_basis(self._xhat, self._yhat)
            except TypeError:
                if self._derivative_order_xhat > 1 or self._derivative_order_yhat > 1:
                    return 0.0
                elif self._basis_idx not in [0, 1, 2]:
                    raise KeyError('Basis index must be in [0, 1, 2]. Given: {}'.format(self._basis_idx))
                else:
                    raise KeyError('Derivative order must be positive.')

        elif self._basis_type == 202:  # quadratic basis
            # Set up cases. Dictionary keys are defined as (basis_idx, derivative_order_xhat, derivative_order_yhat)
            basis_dispatcher = {
                (0, 0, 0): lambda x_hat, y_hat: (2 * x_hat**2 + 2 * y_hat**2 + 4 * x_hat * y_hat - 3 * y_hat
                                                 - 3 * x_hat + 1),  # Psi_hat_0
                (0, 1, 0): lambda x_hat, y_hat: 4 * x_hat + 4 * y_hat - 3,  # diff(Psi_hat_0, x_hat)
                (0, 0, 1): lambda x_hat, y_hat: 4 * y_hat + 4 * x_hat - 3,  # diff(Psi_hat_0, y_hat)
                (0, 1, 1): lambda x_hat, y_hat: 4,  # diff(Psi_hat_0, x_hat, y_hat)
                (0, 2, 0): lambda x_hat, y_hat: 4,  # diff(Psi_hat_0, x_hat$2)
                (0, 0, 2): lambda x_hat, y_hat: 4,  # diff(Psi_hat_0, y_hat$2)

                (1, 0, 0): lambda x_hat, y_hat: 2 * x_hat**2 - x_hat,  # Psi_hat_1
                (1, 1, 0): lambda x_hat, y_hat: 4 * x_hat - 1,  # diff(Psi_hat_1, x_hat)
                (1, 0, 1): lambda x_hat, y_hat: 0,  # diff(Psi_hat_1, y_hat)
                (1, 1, 1): lambda x_hat, y_hat: 0,  # diff(Psi_hat_1, x_hat, y_hat)
                (1, 2, 0): lambda x_hat, y_hat: 4,  # diff(Psi_hat_1, x_hat$2)
                (1, 0, 2): lambda x_hat, y_hat: 0,  # diff(Psi_hat_1, y_hat$2)

                (2, 0, 0): lambda x_hat, y_hat: 2 * y_hat**2 - y_hat,  # Psi_hat_2
                (2, 1, 0): lambda x_hat, y_hat: 0,  # diff(Psi_hat_2, x_hat)
                (2, 0, 1): lambda x_hat, y_hat: 4 * y_hat - 1,  # diff(Psi_hat_2, y_hat)
                (2, 1, 1): lambda x_hat, y_hat: 0,  # diff(Psi_hat_2, x_hat, y_hat)
                (2, 2, 0): lambda x_hat, y_hat: 0,  # diff(Psi_hat_2, x_hat$2)
                (2, 0, 2): lambda x_hat, y_hat: 4,  # diff(Psi_hat_2, y_hat$2)

                (3, 0, 0): lambda x_hat, y_hat: -4 * x_hat**2 - 4 * x_hat * y_hat + 4 * x_hat,  # Psi_hat_3
                (3, 1, 0): lambda x_hat, y_hat: -8 * x_hat - 4 * y_hat + 4,  # diff(Psi_hat_3, x_hat)
                (3, 0, 1): lambda x_hat, y_hat: -4 * x_hat,  # diff(Psi_hat_3, y_hat)
                (3, 1, 1): lambda x_hat, y_hat: -4,  # diff(Psi_hat_3, x_hat, y_hat)
                (3, 2, 0): lambda x_hat, y_hat: -8,  # diff(Psi_hat_3, x_hat$2)
                (3, 0, 2): lambda x_hat, y_hat: -8,  # diff(Psi_hat_3, y_hat$2)

                (4, 0, 0): lambda x_hat, y_hat: 4 * x_hat * y_hat,  # Psi_hat_4
                (4, 1, 0): lambda x_hat, y_hat: 4 * y_hat,  # diff(Psi_hat_4, x_hat)
                (4, 0, 1): lambda x_hat, y_hat: 4 * x_hat,  # diff(Psi_hat_4, y_hat)
                (4, 1, 1): lambda x_hat, y_hat: 4,  # diff(Psi_hat_4, x_hat, y_hat)
                (4, 2, 0): lambda x_hat, y_hat: 0,  # diff(Psi_hat_4, x_hat$2)
                (4, 0, 2): lambda x_hat, y_hat: 0,  # diff(Psi_hat_2, y_hat$2)

                (5, 0, 0): lambda x_hat, y_hat: -4 * y_hat**2 - 4 * x_hat * y_hat + 4 * y_hat,  # Psi_hat_5
                (5, 1, 0): lambda x_hat, y_hat: -4 * y_hat,  # diff(Psi_hat_5, x_hat)
                (5, 0, 1): lambda x_hat, y_hat: -8 * y_hat - 4 * x_hat + 4,  # diff(Psi_hat_5, y_hat)
                (5, 1, 1): lambda x_hat, y_hat: -4,  # diff(Psi_hat_5, x_hat, y_hat)
                (5, 2, 0): lambda x_hat, y_hat: 0,  # diff(Psi_hat_5, x_hat$2)
                (5, 0, 2): lambda x_hat, y_hat: -8  # diff(Psi_hat_5, y_hat$2)
            }

            ref_basis = basis_dispatcher.get((self._basis_idx,
                                              self._derivative_order_xhat,
                                              self._derivative_order_yhat))
            try:
                return ref_basis(self._xhat, self._yhat)
            except TypeError:
                if self._derivative_order_xhat > 2 or self._derivative_order_yhat > 2:
                    return 0.0
                elif self._derivative_order_xhat == 2 and self._derivative_order_yhat > 0:
                    return 0.0
                elif self._derivative_order_xhat > 0 and self._derivative_order_yhat == 2:
                    return 0.0
                elif self._basis_idx not in range(6):
                    raise KeyError('Basis index must be in [0, ..., 5]. Given: {}'.format(self._basis_idx))
                else:
                    raise KeyError('Derivative order must be positive.')

    def fe_local_basis(self, coords, vertices, basis_idx, derivative_order):
        """

        Parameters
        ----------
        coords
        vertices
        basis_idx
        derivative_order

        Returns
        -------

        """
        # Extract our coordinates
        x, y = coords
        # Set the basis index
        self._basis_idx = basis_idx

        # The vertices of the triangular finite elment E_n
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]

        # Determinant of Jacobi matrix
        det_jacobian = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # The affine mapping from the local triangular element with vertices {(x1,y1), (x2,y2), (x3,y3)} to the
        # reference element with vertices {(0,0), (1,0), (0,1)}
        self._xhat = ((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1)) / det_jacobian
        self._yhat = (-(y2 - y1) * (x - x1) + (x2 - x1) * (y - y1)) / det_jacobian

        # Evaluate the reference basis function
        # For derivatives, we have applied the chain rule
        if derivative_order == (0, 0):
            return self.__fe_reference_basis((0, 0))
        elif derivative_order == (1, 0):
            # Compute df/d(x_hat) * d(x_hat)/dx
            ret_value = self.__fe_reference_basis((1, 0)) * (y3 - y1) / det_jacobian

            # Compute df/d(y_hat) * d(y_hat)/dx
            ret_value += self.__fe_reference_basis((0, 1)) * (y1 - y2) / det_jacobian

        elif derivative_order == (0, 1):
            # Compute df/d(x_hat) * d(x_hat)/dy
            ret_value = self.__fe_reference_basis((1, 0)) * (x1 - x3) / det_jacobian

            # Compute df/d(y_hat) * d(y_hat)/dy
            ret_value += self.__fe_reference_basis((0, 1)) * (x2 - x1) / det_jacobian

        elif derivative_order == (1, 1):
            # Compute d2f/d(x_hat)2 * d(x_hat)/dy * d(x_hat)/dx
            ret_value = self.__fe_reference_basis((2, 0)) * (x1 - x3) * (y3 - y1) / det_jacobian**2

            # Compute d2f/d(x_hat)d(y_hat) * d(x_hat)/dy * d(y_hat)/dx
            ret_value += self.__fe_reference_basis((1, 1)) * (x1 - x3) * (y1 - y2) / det_jacobian**2

            # Compute d2f/d(x_hat)d(y_hat) * d(y_hat)/dy * d(x_hat)/dx
            ret_value += self.__fe_reference_basis((1, 1)) * (x2 - x1) * (y3 - y1) / det_jacobian**2

            # Computer d2f/d(y_hat)2 * d(y_hat)/dy * d(y_hat)/dx
            ret_value += self.__fe_reference_basis((0, 2)) * (x2 - x1) * (y1 - y2) / det_jacobian**2

        elif derivative_order == (2, 0):
            # Compute d2f/d(x_hat)2 * d(x_hat)/dx * d(x_hat)/dx
            ret_value = self.__fe_reference_basis((2, 0)) * (y3 - y1)**2 / det_jacobian**2

            # Compute d2f/d(x_hat)d(y_hat) * d(x_hat)/dx * d(y_hat)/dx
            ret_value += 2 * self.__fe_reference_basis((1, 1)) * (y3 - y1) * (y1 - y2) / det_jacobian**2

            # Compute d2f/d(y_hat)2 * d(y_hat)/dx * d(y_hat)/dx
            ret_value += self.__fe_reference_basis((0, 2)) * (y1 - y2)**2 / det_jacobian**2

        elif derivative_order == (0, 2):
            # Compute d2f/d(x_hat)2 * d(x_hat)/dy * d(x_hat)/dy
            ret_value = self.__fe_reference_basis((2, 0)) * (x1 - x3)**2 / det_jacobian**2

            # Compute d2f/d(x_hat)d(y_hat) * d(x_hat)/dy * d(y_hat)/dy
            ret_value += 2 * self.__fe_reference_basis((1, 1)) * (x1 - x3) * (x2 - x1) / det_jacobian**2

            # Compute d2f/d(y_hat)2 * d(y_hat)/dy * d(y_hat)/dy
            ret_value += self.__fe_reference_basis((0, 2)) * (x2 - x1)**2 / det_jacobian**2

        else:
            raise ValueError('Derivative order must be defined from 0 to 2 on each axis.')

        return ret_value


BasisInfo = namedtuple('BasisInfo', 'aka')
_BASIS_TYPE = {
    101: BasisInfo(aka=[101, 'linear']),
    102: BasisInfo(aka=[102, 'quadratic']),
    201: BasisInfo(aka=[201, 'linear2D_tri']),
    202: BasisInfo(aka=[202, 'quadratic2D_tri'])
}

_BASIS_ALIAS = dict((alias, name)
                    for name, info in _BASIS_TYPE.items()
                    for alias in info.aka)
