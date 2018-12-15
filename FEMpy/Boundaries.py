"""
Boundaries.py
Author: Benjamin Floyd

Contains classes to define and treat boundary conditions.
"""

from collections import namedtuple

import numpy as np
from scipy.sparse import lil_matrix

from .helpers import basis_type_parser
from .helpers import line_integral, copy_docstring_from


class BoundaryConditions(object):
    r"""
    Defines all boundary conditions to be applied to a Poisson equation.

    Takes in the coordinates for the boundary nodes and the boundary condition types and provides the treatments for
    each of the boundry condition types.

    Parameters
    ----------
    mesh : {:class: FEMpy.Mesh.Interval1D, :class: FEMpy.Mesh.TriangularMesh2D}
        A :class:`Mesh` class defining the mesh and associated information matrices.
    boundary_types : array_like of str {'dirichlet`, 'neumann', 'robin'}
        Boundary condition type for each coordinate in `boundary_coords`.
    boundary_coords : array_like, optional
        List of coordinate values of the boundary nodes. If not specified, will use the boundary node coordinates stored
        in `mesh`.
    dirichlet_fun : function, optional
        The Dirichlet boundary condition function `g`(`x`). Must be defined at the boundary values specified in
        `boundary_coords`.
    neumann_fun : function, optional
        The Neumann boundary condition function `r`(`x`). Must be defined at the bounadry values specified in
        `boundary_coords`.
    robin_fun_q : function, optional
        The Robin boundary condition function `q`(`x`). Must be defined at the boundary values specified in
        `boundary_coords`.
    robin_fun_p : function, optional
        The Robin boundary condition function `p`(`x`). Must be defined at the boundary values specified in
        `boundary_coords`.
    coeff_fun : function, optional
        Function name of the coefficient function `c`(`x`) in the Poisson equation. Required if Neumann or Robin
        boundary conditions are included.


    .. warning:: Both end point boundary conditions cannot be Neumann as this may result in a loss of uniqueness of the
        solution.

    Notes
    -----
    - The Dirichlet boundary conditions are be defined as

      .. math::

        u(x) = g(x); x = a \text{ or } b.

    - The Neumann boundary conditions are defined as

      .. math::

        \frac{{\rm d}}{{\rm d} x} u(x) = r(x); x = a \text{ or } b.

    - The Robin boundary conditions are defined as

      .. math::

        \frac{{\rm d}}{{\rm d}x} u(x) + q(x) u(x) = p(x); x = a \text{ or } b.

    """

    def __init__(self, mesh, boundary_types, boundary_coords=None,
                 dirichlet_fun=None, neumann_fun=None, robin_fun_q=None, robin_fun_p=None, coeff_fun=None):
        self._mesh = mesh
        self._boundary_node_types = boundary_types

        if boundary_coords is not None:
            self._boundary_node_coords = np.asanyarray(boundary_coords)
        else:
            self._boundary_node_coords = mesh.boundary_nodes.T.copy()

        # Initialize the functions
        self._dirchlet_fun = dirichlet_fun
        self._neumann_fun = neumann_fun
        self._robin_fun_q = robin_fun_q
        self._robin_fun_p = robin_fun_p
        self._coeff_fun = coeff_fun

        self._generate_boundary_nodes()

    def _generate_boundary_nodes(self):
        """Creates the boundary node information matrix."""

        self._boundary_nodes = np.empty((2, len(self._boundary_node_types)), dtype='int64')
        for i in range(len(self._boundary_node_types)):
            if isinstance(self._boundary_node_types[i], int):
                b_type = _BOUNDARY_ALIAS.get(self._boundary_node_types[i], None)
            elif isinstance(self._boundary_node_types[i], str):
                bstr = self._boundary_node_types[i].lower()
                b_type = _BOUNDARY_ALIAS.get(bstr, None)
            else:
                raise TypeError('Boundary Type must be a string identifier or an integer code.')

            self._boundary_nodes[0, i] = b_type

        self._boundary_nodes[1, :] = [np.where(self._mesh.Pb == coord)[0] for coord in self._boundary_node_coords]

    def treat_dirichlet(self, matrix, vector):
        """
        Overwrites the appropriate entries in the stiffness matrix and load vector.

        Corrects the stiffness matrix and load vector to properly incorporate the boundary conditions.

        Parameters
        ----------
        matrix : ndarray
            Finite element stiffness matrix.
        vector : ndarray
            Finite element load vector.
        Returns
        -------
        matrix : ndarray
            Corrected finite element stiffness matrix with Dirichlet boundary conditions incorporated.
        vector : ndarray
            Corrected finite element load vector with Dirichlet boundary conditions incorporated.
        """

        for k in range(self._boundary_nodes.shape[1]):
            if self._boundary_nodes[0, k] == -1:  # Dirichlet boundary conditions
                # Get the finite element node index from the information matrix
                i = int(self._boundary_nodes[1, k])

                # Set the appropriate values in the stiffness matrix according to the boundary condition
                matrix[i, :] = 0
                matrix[i, i] = 1

                # Set the appropriate values in the load vector according to the boundary condition
                vector[i] = self._dirchlet_fun(self._mesh.Pb[..., i])

        return matrix, vector

    def treat_neumann(self, vector):
        """
        Overwrites the appropriate entries in the load vector.

        Corrects the load vector to properly incorporate the boundary conditions.

        Parameters
        ----------
        vector : ndarray
            Finite element load vector.

        Returns
        -------
        vector : ndarray
            Corrected finite element load vector with Neumann boundary conditions incorporated.
        """
        vector = np.copy(vector)
        for k in range(self._boundary_nodes.shape[1]):
            if self._boundary_nodes[0, k] == -2:  # Neumann boundary conditions
                # For the 1D case, if the boundary node is on the left-hand end point we will subtract the boundary
                # condition functions from the appropriate entries in the load vector. Otherwise, the boundary condition
                # is on the right-hand end point and will be added to the appropriate entries.
                if k == 0:
                    sign = -1
                else:
                    sign = 1

                # Get the finite element node index from the information matrix
                i = int(self._boundary_nodes[1, k])

                # Set the appropriate values in our vector according to the boundary conditions
                vector[i] += sign * self._neumann_fun(self._mesh.Pb[i]) * self._coeff_fun(self._mesh.Pb[i])

        return vector

    def treat_robin(self, matrix, vector):
        """
        Overwrites the appropriate entries in the stiffness matrix and load vector.

        Corrects the stiffness matrix and load vector to properly incorporate the boundary conditions.

        Parameters
        ----------
        matrix : ndarray
            Finite element stiffness matrix.
        vector : ndarray
            Finite element load vector.

        Returns
        -------
        matrix : ndarray
            Corrected finite element stiffness matrix with Robin boundary conditions incorporated.
        vector : ndarray
            Corrected finite element load vector with Robin boundary conditions incorporated.
        """

        matrix = matrix.copy()
        vector = np.copy(vector)
        for k in range(self._boundary_nodes.shape[1]):
            if self._boundary_nodes[0, k] == -3:  # Robin boundary conditions
                # If the boundary node is on the left-hand end point we will subtract the boundary condition functions
                # from the appropriate entries in the stiffness matrix or load vector. Otherwise, the boundary condition
                # is on the right-hand end point and will be added to the appropriate entries.
                if k == 0:
                    sign = -1
                else:
                    sign = 1

                # Get the finite element node index form the information matrix
                i = int(self._boundary_nodes[1, k])

                # Set the appropriate values in the matrix according to the boundary conditions
                matrix[i, i] += sign * self._robin_fun_q(self._mesh.Pb[i]) * self._coeff_fun(self._mesh.Pb[i])

                # Set the appropriate values in the vector according to the boundary conditions
                vector[i] += sign * self._robin_fun_p(self._mesh.Pb[i]) * self._coeff_fun(self._mesh.Pb[i])

        return matrix, vector


class BoundaryConditions2D(BoundaryConditions):
    r"""
    Defines all boundary conditions to be applied to a Poisson equation.

    Takes in the coordinates for the boundary nodes and boundary edges and the boundary condition types for each and
    provides the treatments for each of the boundry condition types.

    Parameters
    ----------
    mesh : {:class: FEMpy.Mesh.Interval1D, :class: FEMpy.Mesh.TriangularMesh2D}
        A :class:`Mesh` class defining the mesh and associated information matrices.
    boundary_node_types : array_like of str {'dirichlet`, 'neumann', 'robin'}
        Boundary condition type for each coordinate in `boundary_node_coords`.
    boundary_edge_types : array_like of str {'dirichlet`, 'neumann', 'robin'}
        Boundary condition type for each edge segment in `boundary_edge_coords`.
    boundary_node_coords : array_like, optional
        List of coordinate values of the boundary nodes. If not specified, will use the boundary node coordinates stored
        in `mesh`.
    boundary_edge_coords : array_like, optional
        List of grid coordinates for edge nodes. If not specified, will use the boundary edge coordinates stored in
        `mesh`.
    trial_basis, test_basis : {:class: FEMpy.FEBasis.IntervalBasis1D, :class: FEMpy.FEBasis.TriangularBasis2D}, optional
        A :class: `FEBasis` class defining the finite element basis functions for the trial and test bases. If Neumann
        boundary conditions included `test_basis` is required. If Robin boundary conditions included, then both bases
        are required.
    dirichlet_fun : function, optional
        The Dirichlet boundary condition function `g`(`x`, `y`). Must be defined at the boundary values specified in
        `boundary_coords`.
    neumann_fun : function, optional
        The Neumann boundary condition function `r`(`x`, `y`). Must be defined at the bounadry values specified in
        `boundary_coords`.
    robin_fun_q : function, optional
        The Robin boundary condition function `q`(`x`, `y`). Must be defined at the boundary values specified in
        `boundary_coords`.
    robin_fun_p : function, optional
        The Robin boundary condition function `p`(`x`, `y`). Must be defined at the boundary values specified in
        `boundary_coords`.
    coeff_fun : function, optional
        Function name of the coefficient function `c`(`x`, `y`) in the Poisson equation. Required if Neumann or Robin
        boundary conditions are included.


    .. warning:: All edge boundary conditions cannot be Neumann as this may result in a loss of uniqueness of the
       solution.

    Notes
    -----
    - The Dirichlet boundary conditions are be defined as

      .. math::

        u(x, y) = g(x, y); (x, y) \in \delta\Omega \setminus (\Gamma_1 \cup \Gamma_2).

    - The Neumann boundary conditions are defined as

      .. math::

        \nabla u(x, y) \cdot \hat{\mathbf{n}} = r(x, y); (x, y) \in \Gamma_1 \subseteq \delta\Omega,

    - The Robin boundary conditions are defined as

      .. math::

        \nabla u(x, y) \cdot \hat{\mathbf{n}} + q(x, y) u(x, y) = p(x, y); (x, y) \in \Gamma_2 \subseteq \delta\Omega.

    """

    def __init__(self, mesh, boundary_node_types, boundary_edge_types,
                 boundary_node_coords=None, boundary_edge_coords=None,
                 trial_basis=None, test_basis=None,
                 dirichlet_fun=None, neumann_fun=None, robin_fun_q=None, robin_fun_p=None, coeff_fun=None):
        super().__init__(mesh, boundary_node_types, boundary_node_coords,
                         dirichlet_fun, neumann_fun, robin_fun_q, robin_fun_p, coeff_fun)
        if boundary_edge_coords is not None:
            self._boundary_edge_coords = np.asanyarray(boundary_edge_coords)
        else:
            self._boundary_edge_coords = mesh.boundary_edges.T.copy()

        self._boundary_edge_types = boundary_edge_types
        self._trial_basis = trial_basis
        self._test_basis = test_basis

        # Get the number of basis functions.
        if self._trial_basis is not None:
            self._num_local_trial = basis_type_parser(self._trial_basis.basis_type, self._mesh)[1]
        if self._test_basis is not None:
            self._num_local_test = basis_type_parser(self._test_basis.basis_type, self._mesh)[1]

        # Generate the boundary edge informaiton matrix
        self._generate_boundary_edges()

    def _generate_boundary_nodes(self):
        """Creates the boundary node information matrix."""

        # Initialize the information matrix
        self._boundary_nodes = np.empty((2, len(self._boundary_node_types)), dtype='int64')

        # Iterate through the input boundary types, standardize the type code and store the code in the matrix
        for i in range(len(self._boundary_node_types)):
            if isinstance(self._boundary_node_types[i], np.int64):
                b_type = _BOUNDARY_ALIAS.get(self._boundary_node_types[i], None)
            elif isinstance(self._boundary_node_types[i], str):
                bstr = self._boundary_node_types[i].lower()
                b_type = _BOUNDARY_ALIAS.get(bstr, None)
            else:
                raise TypeError('Boundary Type must be a string identifier or an integer code.')

            self._boundary_nodes[0, i] = b_type

        dtype_nodes = {'names': ['f{}'.format(i) for i in range(self._mesh.Pb.shape[0])],
                       'formats': self._mesh.Pb.shape[0] * [self._mesh.Pb.dtype]}
        for i in range(self._boundary_nodes.shape[1]):
            self._boundary_nodes[1, i] = np.intersect1d(self._mesh.Pb.T.copy().view(dtype_nodes),
                                                    self._boundary_node_coords.view(dtype_nodes)[i,:],
                                                    return_indices=True)[1]

    def _generate_boundary_edges(self):
        """Creates the boundary edge information matrix."""

        # Initialize the information matrix
        self._boundary_edges = np.empty((4, len(self._boundary_edge_types)), dtype='int64')

        # Iterate through the input boundary types, standardize the type code and store the code in the matrix
        for i in range(len(self._boundary_edge_types)):
            if isinstance(self._boundary_edge_types[i], np.int64):
                b_type = _BOUNDARY_ALIAS.get(self._boundary_edge_types[i], None)
            elif isinstance(self._boundary_edge_types[i], str):
                bstr = self._boundary_edge_types[i].lower()
                b_type = _BOUNDARY_ALIAS.get(bstr, None)
            else:
                raise TypeError('Boundary Type must be a string identifier or an integer code.')

            self._boundary_edges[0, i] = b_type

        # Find the node index of the input edge coordinates using the P matrix
        dtype_edges = {'names': ['f{}'.format(i) for i in range(self._mesh.P.shape[0])],
                       'formats': self._mesh.P.shape[0] * [self._mesh.P.dtype]}
        node_idx = np.empty(self._boundary_edge_coords.shape[0])
        for i in range(len(node_idx)):
            node_idx[i] = np.intersect1d(self._mesh.P.T.copy().view(dtype_edges),
                                  self._boundary_edge_coords[i,:].view(dtype_edges),
                                  return_indices=True)[1]
        # node_idx = node_idx[np.newaxis].T

        # Search in the T matrix for the column containing the node indices for each edge
        for i in range(len(node_idx)-1):
            # The edge nodes are adjacent node indices in our `node_idx` vector
            edge_nodes = node_idx[i:i+2]

            # Check the membership of the entries of `edge nodes` in T.
            # The column of T containing both entries will be the finite element index associated with the edge end
            # points in `edge nodes`.
            element_idx = np.where(np.sum(np.isin(self._mesh.T, edge_nodes), axis=0) == 2)[0]

            # Store the finite element index and edge end points into the matrix
            self._boundary_edges[1, i] = element_idx
            self._boundary_edges[2:4, i] = edge_nodes

    @copy_docstring_from(BoundaryConditions.treat_neumann)
    def treat_neumann(self, vector):
        vector = np.copy(vector)
        # Initialize the Neumann vector `v`
        v = np.zeros_like(vector)

        for k in range(self._boundary_edges.shape[1]):
            if self._boundary_edges[0, k] == -2:  # Neumann boundary conditions
                # Get the element index associated with the kth edge
                element_idx = int(self._boundary_edges[1, k])

                # Extract the global node coordinates to evaluate our integral on
                vertices = self._mesh.get_vertices(element_idx).T

                # Extract the global node coordinates of the edge end points to evaluate our integral on
                edge_endpts = self._mesh.P[:, (self._boundary_edges[2:4, k])]

                for beta in range(self._num_local_test):
                    def integrand_v(coords):
                        return (self._coeff_fun(coords) * self._neumann_fun(coords)
                                * self._test_basis.fe_local_basis(coords, vertices=vertices, basis_idx=beta,
                                                                  derivative_order=(0, 0)))

                    # Compute the line integral along the edge
                    int_value_v = line_integral(integrand_v, edge_endpts)

                    # Store our integral value in `v`
                    v[self._mesh.Tb[beta, element_idx]] += int_value_v

        # Apply the treatment to the load vector
        vector += v

        return vector

    @copy_docstring_from(BoundaryConditions.treat_robin)
    def treat_robin(self, matrix, vector):
        matrix = matrix.copy()
        vector = np.copy(vector)
        # The number of boundary edges
        num_boundary_edges = self._boundary_edges.shape[1]

        # Initialize the Robin vector `w` and matrix `r`
        w = np.zeros_like(vector)
        r = lil_matrix(matrix.shape)

        for k in range(num_boundary_edges):
            if self._boundary_edges[0, k] == -3:  # Robin boundary conditions
                # Get the element index associated with the kth edge
                element_idx = int(self._boundary_edges[1, k])

                # Extract the global node coordinates to evaluate our integral on
                vertices = self._mesh.get_vertices(element_idx).T

                # Extract the global node coordinates of the edge end points to evaluate our integral on
                edge_endpts = self._mesh.P[:, self._boundary_edges[2:4, k]]

                for beta in range(self._num_local_test):
                    def integrand_w(coords):
                        return (self._coeff_fun(coords) * self._robin_fun_p(coords)
                                * self._test_basis.fe_local_basis(coords, vertices=vertices, basis_idx=beta,
                                                                  derivative_order=(0, 0)))

                    # Compute the line integral along the edge
                    int_value_w = line_integral(integrand_w, edge_endpts)

                    # Store our integral value in `w`
                    w[self._mesh.Tb[beta, element_idx]] += int_value_w

                    for alpha in range(self._num_local_trial):
                        def integrand_r(coords):
                            return (self._coeff_fun(coords) * self._robin_fun_q(coords)
                                    * self._test_basis.fe_local_basis(coords, vertices=vertices, basis_idx=beta,
                                                                      derivative_order=(0, 0))
                                    * self._trial_basis.fe_local_basis(coords, vertices=vertices, basis_idx=alpha,
                                                                       derivative_order=(0, 0)))

                        # Compute the line integral along the edge
                        int_value_r = line_integral(integrand_r, edge_endpts)

                        # Store our integral value in `r`
                        r[self._mesh.Tb[beta, element_idx], self._mesh.Tb[alpha, element_idx]] += int_value_r

        # Apply the treatment to the matrix and vector
        matrix += r.tocsr()
        vector += w

        return matrix, vector


BoundaryInfo = namedtuple('BoundaryInfo', 'aka')
_BOUNDARY_TYPE = {
    -1: BoundaryInfo(aka=[-1, 'dirichlet', 'dir', 'd' 'g']),
    -2: BoundaryInfo(aka=[-2, 'neumann', 'neu', 'n', 'r']),
    -3: BoundaryInfo(aka=[-3, 'robin', 'rob', 'pq', 'p', 'q'])
}

_BOUNDARY_ALIAS = dict((alias, name)
                       for name, info in _BOUNDARY_TYPE.items()
                       for alias in info.aka)
