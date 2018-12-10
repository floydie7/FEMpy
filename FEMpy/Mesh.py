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
    Defines a one-dimensional mesh and associated information matrices.

    Create a mesh object defined from `left` to `right` with step size `h`. This object provides the information
    matrices describing the mesh (`P` and `T`) as well as the information matrices describing the finite element of type
    `basis_type` (`Pb` and `Tb`).

    Parameters
    ----------
    left : float
        The left end point of the domain.
    right : float
        The right end point of the domain.
    h : float
        Mesh grid spacing.
    basis_type : {101, 'linear', 102, 'quadratic'}
        Finite element basis type. Can either be called as a integer code or a string identifier to indicate the
        type of basis function we are using.

        - 101, 'linear' : 1-dimensional, linear basis.
        - 102, 'quadratic' : 1-dimensional, quadratic basis.

    Attributes
    ----------
    P : ndarray
        Information matrix containing the coordinates of all mesh nodes.
    T : ndarray
        Information matrix containing the global node indices of the mesh nodes of all the mesh elements.
    Pb : ndarray
        Information matrix containing the coordinates of all finite element nodes.
    Tb : ndarray
        Information matrix containing the global node indices of the finite element nodes.

    Examples
    --------
    Automatically generate the boundary nodes.
    >>> from FEMpy import Interval1D
    >>> mesh = Interval1D(0, 1, 1/2, 'linear')
    >>> mesh.boundary_nodes
    >>> array([0., 1.])

    Get the vertices of the nth element.
    >>> from FEMpy import Interval1D
    >>> mesh = Interval1D(0, 1, 1/4, 'quadratic')
    >>> mesh.get_vertices(3)
    >>> array([0.75, 1.  ])

    Show the number of elements in the interval.
    >>> from FEMpy import Interval1D
    >>> mesh = Interval1D(-1, 3, 1/8, 'quadratic')
    >>> mesh.num_elements_x
    >>> 32
    """

    def __init__(self, left, right, h, basis_type):
        self._left = float(left)
        self._right = float(right)
        self._h1 = float(h)
        self._num_elements_x = int((self._right - self._left) / self._h1)

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
        self._generate_p_t_matrices()
        self._generate_pb_tb_matrices()
        self._generate_boundary_node_coords()

    @property
    def num_elements_x(self):
        """Returns the number of elements along the domain x-axis."""
        return self._num_elements_x

    @property
    def boundary_nodes(self):
        """Returns the boundary node coordinates generated from the mesh."""
        return self._boundary_node_coords

    def get_vertices(self, n):
        """
        Extracts the vertices of the mesh element `n`.

        Parameters
        ----------
        n : int
            Mesh element index.

        Returns
        -------
        ndarray
            The vertices of the mesh element `En`.
        """
        return self.P[..., self.T[:, n]]

    def _generate_p_t_matrices(self):
        """Generates the mesh node coordinate matrix `P` and the mesh node index matrix `T`."""
        self.P = np.linspace(self._left, self._right, num=int(self._num_elements_x + 1))
        self.T = np.array([np.arange(0, self._num_elements_x), np.arange(1, self._num_elements_x + 1)], dtype='int64')

    def _generate_pb_tb_matrices(self):
        """Generates the finite element coordinate matrix `Pb` and the finite element index matrix `Tb`."""
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

    def _generate_boundary_node_coords(self):
        """Generates the boundary edge coordinates based on the mesh."""
        self._boundary_node_coords = np.array([self._left, self._right])


class TriangularMesh2D(Interval1D):
    """
    Defines a two-dimensional mesh with triangular elements and associated information matrices.

    Create a mesh object defined on the domain [`left`, `right] x [`bottom`, `top`] with step size `h1` in the
    x-direction and `h2` in the y-direction. This object provides the information matrices describing the mesh
    (`P` and `T`) as well as the information matrices describing the finite element of type `basis_type`
    (`Pb` and `Tb`).

    Parameters
    ----------
    left : float
        The left edge point of the domain.
    right : float
        The right edge point of the domain.
    bottom : float
        The bottom edge point of the domain.
    top : float
        The top edge point of the domain.
    h1 : float
        Mesh grid spacing along x-direction.
    h2 : float
        Mesh grid spacing along y-direction.
    basis_type : {201, 'linear', 'linear2D_tri', 202, 'quadratic', 'quadratic2D_tri'}
        Finite element basis type. Can either be called as a integer code or a string identifier to indicate the
        type of basis function we are using.

        - 201, 'linear', 'linear2D_tri : 1-dimensional, linear basis on triangular elements.
        - 202, 'quadratic', 'quadratic2D_tri : 1-dimensional, quadratic basis on triangular elements.

    Attributes
    ----------
    P : ndarray
        Information matrix containing the coordinates of all mesh nodes.
    T : ndarray
        Information matrix containing the global node indices of the mesh nodes of all the mesh elements.
    Pb : ndarray
        Information matrix containing the coordinates of all finite element nodes.
    Tb : ndarray
        Information matrix containing the global node indices of the finite element nodes.

    Examples
    --------
    Automatically generate the boundary nodes and edges.
    >>> from FEMpy import TriangularMesh2D
    >>> mesh = TriangularMesh2D(0, 1, 0, 1, 1/2, 1/2, 'quadratic')
    >>> mesh.boundary_nodes
    >>> array([[0.  , 0.25, 0.5 , 0.75, 1., 1.  , 1. , 1.  , 1., 0.75, 0.5, 0.25, 0.  , 0.  , 0.  , 0.  ],
    >>>        [0.  , 0.  , 0.  , 0.  , 0., 0.25, 0.5, 0.75, 1., 1.  , 1. , 1.  , 1.  , 0.75, 0.5 , 0.25]])
    >>> mesh.boundary_edges
    >>> array([[0. , 0.5, 1. , 1. , 1. , 0.5, 0. , 0. ],
    >>>        [0. , 0. , 0. , 0.5, 1. , 1. , 1. , 0.5]])

    Get the number of elements in the x-axis and y-axis.
    >>> from FEMpy import TriangularMesh2D
    >>> mesh = TriangularMesh2D(0, 2, 0, 1, 1/2, 1/2, 'linear')
    >>> mesh.num_elements_x
    >>> 4
    >>> mesh.num_elements_y
    >>> 2
    """

    def __init__(self, left, right, bottom, top, h1, h2, basis_type):
        # Make a quick adjustment to the basis type string identifiers to make them unique
        if isinstance(basis_type, str) and not basis_type.endswith('2D_tri'):
            basis_type += '2D_tri'

        self._bottom = float(bottom)
        self._top = float(top)
        self._h2 = float(h2)
        self._num_elements_y = int((self._top - self._bottom) / self._h2)
        self._num_mesh_nodes = 3

        super().__init__(left, right, h1, basis_type)

        # Generate the boundary node and edge coordinates from the mesh
        self._generate_boundary_node_coords()
        self._generate_boundrary_edge_coords()

    @property
    def num_elements_y(self):
        """Returns the number of elements along the domain y-axis."""
        return self._num_elements_y

    @property
    def boundary_edges(self):
        """Returns the boundary edge coordinates generated from the mesh."""
        return self._boundary_edge_coords

    def _generate_p_t_matrices(self):
        """Generates the mesh node coordinate matrix `P` and the mesh node index matrix `T`."""
        # Compute total number of mesh nodes
        _N_mesh = (self._num_elements_x + 1) * (self._num_elements_y + 1)

        # Generate the mesh coordinates
        yy, xx = np.meshgrid(np.linspace(self._bottom, self._top, num=self._num_elements_y+1),
                             np.linspace(self._left, self._right, num=self._num_elements_x+1),
                             sparse=False, indexing='xy')
        self.P = np.vstack([xx.reshape(-1), yy.reshape(-1)])

        # Initialize T matrix
        self.T = np.empty((3, int(2 * self._num_elements_x * self._num_elements_y)), dtype='int64')
        for row, col in product(range(int(self._num_elements_y)), range(int(self._num_elements_x))):
            # Compute the finite element index for the lower and upper triangular elements within each cell of elements
            lower_element_idx = int(col * 2 * self._num_elements_y + 2 * row)
            upper_element_idx = int(col * 2 * self._num_elements_y + 2 * row + 1)

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

    def _generate_pb_tb_matrices(self):
        """Generates the finite element coordinate matrix `Pb` and the finite element index matrix `Tb`."""
        if self._basis_type == 201:  # linear basis
            self.Pb = self.P
            self.Tb = self.T
        elif self._basis_type == 202:  # quadratic basis
            # Set the number of local nodes based on the basis type
            num_local_nodes = 6

            # Generate the mesh coordinates
            yy, xx = np.meshgrid(np.linspace(self._bottom, self._top, num=2 * self._num_elements_y + 1),
                                 np.linspace(self._left, self._right, num=2 * self._num_elements_x + 1),
                                 sparse=False, indexing='xy')
            self.Pb = np.vstack([xx.reshape(-1), yy.reshape(-1)])

            # Initialize Tb matrix
            self.Tb = np.empty((num_local_nodes, int(2 * self._num_elements_x * self._num_elements_y)), dtype='int64')
            for row, col in product(range(int(self._num_elements_y)), range(int(self._num_elements_x))):
                # Compute the finite element index for the lower and upper triangular elements within
                # each cell of elements
                lower_element_idx = int(col * 2 * self._num_elements_y + 2 * row)
                upper_element_idx = int(col * 2 * self._num_elements_y + 2 * row + 1)

                # Set up switch dictionary
                alpha_dict = {
                    0: {'lower_row_idx': row + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 3 + (upper_element_idx - 2) / 2,
                        'upper_col_idx': col + 1},
                    1: {'lower_row_idx': row + 4 * self._num_elements_y + 2 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 4 * self._num_elements_y + 2 + (upper_element_idx - 1) / 2,
                        'upper_col_idx': col + 1},
                    2: {'lower_row_idx': row + 2 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 4 * self._num_elements_y + 4 + (upper_element_idx - 1) / 2,
                        'upper_col_idx': col + 1},
                    3: {'lower_row_idx': row + 2 * self._num_elements_y + 1 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 2 * self._num_elements_y + 2 + (upper_element_idx - 1) / 2,
                        'upper_col_idx': col + 1},
                    4: {'lower_row_idx': row + 2 * self._num_elements_y + 2 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 4 * self._num_elements_y + 4 + (upper_element_idx - 2) / 2,
                        'upper_col_idx': col + 1},
                    5: {'lower_row_idx': row + 1 + lower_element_idx / 2,
                        'lower_col_idx': col + 1,
                        'upper_row_idx': row + 2 * self._num_elements_y + 4 + (upper_element_idx - 2) / 2,
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

    def _generate_boundary_node_coords(self):
        """Generates the boundary node coordinates based on the mesh."""
        # Set the step sizes based on the basis type
        if self._basis_type == 201:
            step_x = self._h1
            step_y = self._h2
        elif self._basis_type == 202:
            step_x = self._h1 / 2
            step_y = self._h2 / 2
        else:
            raise ValueError('Unknown basis type.')

        # Generate the coordinates along the domain edges
        bottom_x = np.arange(self._left, self._right + step_x, step=step_x)
        bottom_y = np.full_like(bottom_x, self._bottom)

        right_y = np.arange(self._bottom + step_y, self._top + step_y, step=step_y)
        right_x = np.full_like(right_y, self._right)

        top_x = np.arange(self._right - step_x, self._left - step_x, step=-step_x)
        top_y = np.full_like(top_x, self._top)

        left_y = np.arange(self._top - step_y, self._bottom, step=-step_y)
        left_x = np.full_like(left_y, self._left)

        x_coords = np.hstack([bottom_x, right_x, top_x, left_x])
        y_coords = np.hstack([bottom_y, right_y, top_y, left_y])

        self._boundary_node_coords = np.vstack([x_coords, y_coords])

    def _generate_boundrary_edge_coords(self):
        """Generates the boundary edge coordinates based on the mesh."""
        # Set our step size
        step_x = self._h1
        step_y = self._h2

        # Generate the coordinates along the domain edges
        bottom_x = np.arange(self._left, self._right + step_x, step=step_x)
        bottom_y = np.full_like(bottom_x, self._bottom)

        right_y = np.arange(self._bottom + step_y, self._top + step_y, step=step_y)
        right_x = np.full_like(right_y, self._right)

        top_x = np.arange(self._right - step_x, self._left - step_x, step=-step_x)
        top_y = np.full_like(top_x, self._top)

        left_y = np.arange(self._top - step_y, self._bottom, step=-step_y)
        left_x = np.full_like(left_y, self._left)

        x_coords = np.hstack([bottom_x, right_x, top_x, left_x])
        y_coords = np.hstack([bottom_y, right_y, top_y, left_y])

        self._boundary_edge_coords = np.vstack([x_coords, y_coords])


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
