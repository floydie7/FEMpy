"""
Mesh.py
Author: Benjamin Floyd

Generates all mesh information matrices.
"""

from collections import namedtuple
from itertools import product

import numpy as np


class Interval1D(object):
    """

    Parameters
    ----------
    left : float
        The left end point of the domain.
    right : float
        The right end point of the domain.
    h : float

    basis_type
    """

    def __init__(self, left, right, h, basis_type):
        self._left = left
        self._right = right
        self._h1 = h
        self._num_elements_x = (self._right - self._left) / self._h1

        if isinstance(basis_type, int):
            self._basis_type = _BASIS_ALIAS.get(basis_type, None)
        elif isinstance(basis_type, str):
            bstr = basis_type.lower()
            self._basis_type = _BASIS_ALIAS.get(bstr, None)
        else:
            raise TypeError('Basis type must be a string identifier or an integer code.')

        # Initialize the information matrices.
        # These will be updated when the user calls the generating functions.
        self.P = None
        self.T = None
        self.Pb = None
        self.Tb = None

        # Call the generators automatically
        self.generate_p_t_matrices()
        self.generate_pb_tb_matrices()

    @property
    def num_elements_x(self):
        return self._num_elements_x

    def get_vertices(self, n):
        return self.P[..., self.T[:, n]]

    def generate_p_t_matrices(self):
        self.P = np.linspace(self._left, self._right, num=int(self._num_elements_x + 1))
        self.T = np.array([np.arange(0, self._num_elements_x), np.arange(1, self._num_elements_x + 1)], dtype='int64')

        # return self.p, self.t

    def generate_pb_tb_matrices(self):
        if self._basis_type == 101:  # linear basis
            self.Pb = self.P
            self.Tb = self.T
        elif self._basis_type == 102:  # quadratic basis
            num_basis_fns = 2 * self._num_elements_x + 1

            self.Pb = np.array([self._left + (k - 1) * self._h1 / 2 for k in np.arange(1, num_basis_fns + 1)])
            self.Tb = np.array([
                [2 * n - 1 for n in np.arange(1, self._num_elements_x + 1)],
                [2 * n + 1 for n in np.arange(1, self._num_elements_x + 1)],
                [2 * n for n in np.arange(1, self._num_elements_x + 1)]
            ], dtype='int64')
        else:
            raise ValueError('Parameter: basis_type not recognized.')

        # return self.pb, self.tb


class TriangularMesh2D(Interval1D):
    """

    Parameters
    ----------
    left
    right
    bottom
    top
    h1
    h2
    basis_type
    """

    def __init__(self, left, right, bottom, top, h1, h2, basis_type):
        # Make a quick adjustment to the basis type string identifiers to make them unique
        if isinstance(basis_type, str) and not basis_type.endswith('2D_tri'):
            basis_type += '2D_tri'

        self._bottom = bottom
        self._top = top
        self._h2 = h2
        self._num_elements_y = (self._top - self._bottom) / self._h2
        self._num_mesh_nodes = 3

        super().__init__(left, right, h1, basis_type)

    @property
    def num_elements_y(self):
        return self._num_elements_y

    def generate_p_t_matrices(self):
        # Compute total number of mesh nodes
        _N_mesh = (self._num_elements_x + 1) * (self._num_elements_y + 1)

        # Generate the mesh coordinates
        yy, xx = np.meshgrid(np.linspace(self._bottom, self._top, num=self._num_elements_x+1),
                             np.linspace(self._left, self._right, num=self._num_elements_y+1),
                             sparse=False, indexing='xy')
        self.P = np.vstack([xx.reshape(-1), yy.reshape(-1)])

        # Initialize T matrix
        self.T = np.empty((3, int(2 * self._num_elements_x * self._num_elements_y)), dtype='int64')
        for row, col in product(range(int(self._num_elements_x)), range(int(self._num_elements_y))):
            # Compute the finite element index for the lower and upper triangular elements within each cell of elements
            lower_element_idx = int(col * 2 * self._num_elements_x + 2 * row)
            upper_element_idx = int(col * 2 * self._num_elements_x + 2 * row + 1)

            # Set up switch dictionary
            alpha_dict = {
                0: {'lower_row_idx': row,
                    'lower_col_idx': col + 1,
                    'upper_row_idx': row + 1,
                    'upper_col_idx': col + 1},
                1: {'lower_row_idx': row,
                    'lower_col_idx': col + 2,
                    'upper_row_idx': row,
                    'upper_col_idx': col + 2},
                2: {'lower_row_idx': row + 1,
                    'lower_col_idx': col + 1,
                    'upper_row_idx': row + 1,
                    'upper_col_idx': col + 2}
            }

            for alpha in range(int(self._num_mesh_nodes)):
                # Compute the global node index of the lower and upper element for the appropriate level of alpha
                lower_node_idx = ((alpha_dict[alpha]['lower_col_idx'] - 1) * (self._num_elements_y + 1) +
                                  alpha_dict[alpha]['lower_row_idx'])
                upper_node_idx = ((alpha_dict[alpha]['upper_col_idx'] - 1) * (self._num_elements_y + 1) +
                                  alpha_dict[alpha]['upper_row_idx'])

                self.T[alpha, lower_element_idx] = lower_node_idx
                self.T[alpha, upper_element_idx] = upper_node_idx

    def generate_pb_tb_matrices(self):
        if self._basis_type == 201:  # linear basis
            self.Pb = self.P
            self.Tb = self.T
        elif self._basis_type == 202:  # quadratic basis
            # Set the number of local nodes based on the basis type
            num_local_nodes = 6

            # Generate the mesh coordinates
            yy, xx = np.meshgrid(np.linspace(self._bottom, self._top, num=2 * self._num_elements_x + 1),
                                 np.linspace(self._left, self._right, num=2 * self._num_elements_y + 1),
                                 sparse=False, indexing='xy')
            self.Pb = np.vstack([xx.reshape(-1), yy.reshape(-1)])

            # Initialize Tb matrix
            self.Tb = np.empty((num_local_nodes, int(2 * self._num_elements_x * self._num_elements_y)), dtype='int64')
            for row, col in product(range(int(self._num_elements_x)), range(int(self._num_elements_y))):
                # Compute the finite element index for the lower and upper triangular elements within
                # each cell of elements
                lower_element_idx = int(col * 2 * self._num_elements_x + 2 * row)
                upper_element_idx = int(col * 2 * self._num_elements_x + 2 * row + 1)

                # Set up switch dictionary
                alpha_dict = {
                    0: {'lower_row_idx': row + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 3 + (upper_element_idx - 2) / 2,
                        'upper_col_idx': col + 1},
                    1: {'lower_row_idx': row + 4 * self._num_elements_x + 2 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 4 * self._num_elements_x + 2 + (upper_element_idx - 1) / 2,
                        'upper_col_idx': col + 1},
                    2: {'lower_row_idx': row + 2 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 4 * self._num_elements_x + 4 + (upper_element_idx - 1) / 2,
                        'upper_col_idx': col + 1},
                    3: {'lower_row_idx': row + 2 + self._num_elements_x + 1 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 2 * self._num_elements_x + 2 + (upper_element_idx - 1) / 2,
                        'upper_col_idx': col + 1},
                    4: {'lower_row_idx': row + 2 + self._num_elements_x + 2 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 4 * self._num_elements_x + 4 + (upper_element_idx - 2) / 2,
                        'upper_col_idx': col + 1},
                    5: {'lower_row_idx': row + 1 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 2 * self._num_elements_x + 4 + (upper_element_idx - 2) / 2,
                        'upper_col_idx': col + 1}
                }

                for alpha in range(num_local_nodes):
                    # Compute the global node index of the lower and upper element for the appropriate level of alpha
                    lower_node_idx = ((alpha_dict[alpha]['lower_col_idx'] - 1) * (3 * self._num_elements_y + 2) +
                                      alpha_dict[alpha]['lower_row_idx'])
                    upper_node_idx = ((alpha_dict[alpha]['upper_col_idx'] - 1) * (3 * self._num_elements_y + 2) +
                                      alpha_dict[alpha]['upper_row_idx'])

                    self.Tb[alpha, lower_element_idx] = lower_node_idx
                    self.Tb[alpha, upper_element_idx] = upper_node_idx


BasisInfo = namedtuple('BasisInfo', 'aka')
_BASIS_TYPE = {
    101: BasisInfo(aka=[101, 'linear']),
    102: BasisInfo(aka=[102, 'quadratic']),
    201: BasisInfo(aka=[201, 'linear2d_tri']),
    202: BasisInfo(aka=[202, 'quadratic2d_tri'])
}

_BASIS_ALIAS = dict((alias, name)
                    for name, info in _BASIS_TYPE.items()
                    for alias in info.aka)
