"""
Boundaries.py
Author: Benjamin Floyd

Contains classes to define and treat boundary conditions.
"""
from collections import namedtuple

import numpy as np
from scipy.sparse import lil_matrix

from .Assemblers import _basis_type_parser
from .helpers import line_integral


class BoundaryConditions(object):
    def __init__(self, boundary_coords, boundary_types, mesh,
                 dirichlet_fun=None, neumann_fun=None, robin_fun_q=None, robin_fun_p=None, coeff_fun=None):
        self._boundary_node_coords = np.asanyarray(boundary_coords)
        self._boundary_node_types = boundary_types
        self._mesh = mesh

        # Initialize the functions
        self._dirchlet_fun = dirichlet_fun
        self._neumann_fun = neumann_fun
        self._robin_fun_q = robin_fun_q
        self._robin_fun_p = robin_fun_p
        self._coeff_fun = coeff_fun

        self._generate_boundary_nodes()

    def _generate_boundary_nodes(self):
        self._boundary_nodes = np.empty((2, len(self._boundary_node_types)))
        for i in range(len(self._boundary_node_types)):
            if isinstance(self._boundary_node_types[i], int):
                b_type = _BOUNDARY_ALIAS.get(self._boundary_node_types[i], None)
            elif isinstance(self._boundary_node_types, str):
                bstr = self._boundary_node_types[i].lower()
                b_type = _BOUNDARY_ALIAS.get(bstr, None)
            else:
                raise TypeError('Boundary Type must be a string identifier or an integer code.')

            self._boundary_nodes[0, i] = b_type

        self._boundary_nodes[1, :] = [np.where(self._mesh.Pb == coord) for coord in self._boundary_node_coords]

    def treat_dirichlet(self, matrix, vector):
        for k in range(self._boundary_nodes.shape[1]):
            if self._boundary_nodes[1, k] == -1:  # Dirichlet boundary conditions
                # Get the finite element node index from the information matrix
                i = self._boundary_nodes[2, k]

                # Set the appropriate values in the stiffness matrix according to the boundary condition
                matrix[i, :] = 0
                matrix[i, i] = 1

                # Set the appropriate values in the load vector according to the boundary condition
                vector[i] = self._dirchlet_fun(self._mesh.Pb[..., i])

        return matrix, vector

    def treat_neumann(self, vector):
        for k in range(self._boundary_nodes.shape[1]):
            if self._boundary_nodes[1, k] == -2:  # Neumann boundary conditions
                # For the 1D case, if the boundary node is on the left-hand end point we will subtract the boundary
                # condition functions from the appropriate entries in the load vector. Otherwise, the boundary condition
                # is on the right-hand end point and will be added to the appropriate entries.
                if k == 1:
                    sign = -1
                else:
                    sign = 1

                # Get the finite element node index from the information matrix
                i = self._boundary_nodes[2, k]

                # Set the appropriate values in our vector according to the boundary conditions
                vector[i] += sign * self._neumann_fun(self._mesh.Pb[i]) * self._coeff_fun(self._mesh.Pb[i])

        return vector

    def treat_robin(self, matrix, vector):
        for k in range(self._boundary_nodes.shape[1]):
            if self._boundary_nodes[1, k] == -3:  # Robin boundary conditions
                # If the boundary node is on the left-hand end point we will subtract the boundary condition functions
                # from the appropriate entries in the stiffness matrix or load vector. Otherwise, the boundary condition
                # is on the right-hand end point and will be added to the appropriate entries.
                if k == 1:
                    sign = -1
                else:
                    sign = 1

                # Get the finite element node index form the information matrix
                i = self._boundary_nodes[2, k]

                # Set the appropriate values in the matrix according to the boundary conditions
                matrix[i, i] += sign * self._robin_fun_q(self._mesh.Pb[i]) * self._coeff_fun(self._mesh.Pb[i])

                # Set the appropriate values in the vector according to the boundary conditions
                vector[i] += sign * self._robin_fun_p(self._mesh.Pb[i]) * self._coeff_fun(self._mesh.Pb[i])

        return matrix, vector


class BoundaryConditions2D(BoundaryConditions):
    def __init__(self, boundary_node_coords, boundary_node_types,
                 boundary_edge_coords, boundary_edge_types,
                 mesh, trial_basis, test_basis,
                 dirichlet_fun=None, neumann_fun=None, robin_fun_q=None, robin_fun_p=None, coeff_fun=None):
        super().__init__(boundary_node_coords, boundary_node_types, mesh,
                         dirichlet_fun, neumann_fun, robin_fun_q, robin_fun_p, coeff_fun)

        self._boundary_edge_coords = np.asanyarray(boundary_edge_coords)
        self._boundary_edge_types = boundary_edge_types
        self._trial_basis = trial_basis
        self._test_basis = test_basis

        self._num_local_trial = _basis_type_parser(self._trial_basis, self._mesh)[1]
        self._num_local_test = _basis_type_parser(self._test_basis, self._mesh)[1]

        self._generate_boundary_edges()

    def _generate_boundary_nodes(self):
        # Initialize the information matrix
        self._boundary_nodes = np.empty((2, len(self._boundary_node_types)))

        # Iterate through the input boundary types, standardize the type code and store the code in the matrix
        for i in range(len(self._boundary_node_types)):
            if isinstance(self._boundary_node_types[i], int):
                b_type = _BOUNDARY_ALIAS.get(self._boundary_node_types[i], None)
            elif isinstance(self._boundary_node_types, str):
                bstr = self._boundary_node_types[i].lower()
                b_type = _BOUNDARY_ALIAS.get(bstr, None)
            else:
                raise TypeError('Boundary Type must be a string identifier or an integer code.')

            self._boundary_nodes[0, i] = b_type

        dtype_nodes = {'names': ['f{}'.format(i) for i in range(self._mesh.Pb.shape[0])],
                       'formats': self._mesh.Pb.shape[0] * [self._mesh.Pb.dtype]}
        self._boundary_nodes[1, :] = np.intersect1d(self._mesh.Pb.T.copy().view(dtype_nodes),
                                                    self._boundary_node_coords.view(dtype_nodes),
                                                    return_indices=True)[1]

    def _generate_boundary_edges(self):
        # Initialize the information matrix
        self._boundary_edges = np.empty((4, len(self._boundary_edge_types)))

        # Iterate through the input boundary types, standardize the type code and store the code in the matrix
        for i in range(len(self._boundary_edge_types)):
            if isinstance(self._boundary_edge_types[i], int):
                b_type = _BOUNDARY_ALIAS.get(self._boundary_edge_types[i], None)
            elif isinstance(self._boundary_edge_types, str):
                bstr = self._boundary_edge_types[i].lower()
                b_type = _BOUNDARY_ALIAS.get(bstr, None)
            else:
                raise TypeError('Boundary Type must be a string identifier or an integer code.')

            self._boundary_edges[0, i] = b_type

        # Find the node index of the input edge coordinates using the P matrix
        dtype_edges = {'names': ['f{}'.format(i) for i in range(self._mesh.P.shape[0])],
                       'formats': self._mesh.P.shape[0] * [self._mesh.P.dtype]}
        node_idx = np.intersect1d(self._mesh.P.T.copy().view(dtype_edges),
                                  self._boundary_edge_coords.view(dtype_edges),
                                  return_indices=True)[1]
        node_idx = node_idx[np.newaxis].T

        # Search in the T matrix for the column containing the node indices for each edge
        for i in range(len(node_idx)-1):
            # The edge nodes are adjacent node indices in our `node_idx` vector
            edge_nodes = node_idx[i:i+1]

            # Check the membership of the entries of `edge nodes` in T.
            # The column of T containing both entries will be the finite element index associated with the edge end
            # points in `edge nodes`.
            element_idx = np.where(np.sum(np.isin(self._mesh.T, edge_nodes), axis=0) == 2)

            # Store the finite element index and edge end points into the matrix
            self._boundary_edges[1, i] = element_idx
            self._boundary_edges[2:3, i] = edge_nodes

    def treat_neumann(self, vector):
        # Initialize the Neumann vector `v`
        v = np.zeros_like(vector)

        for k in range(len(vector)):
            if self._boundary_edges[1, k] == -2:  # Neumann boundary conditions
                # Get the element index associated with the kth edge
                element_idx = self._boundary_edges[2, k]

                # Extract the global node coordinates to evaluate our integral on
                vertices = self._mesh.get_vertices(element_idx)

                # Extract the global node coordinates of the edge end points to evaluate our integral on
                edge_endpts = self._mesh.P[:, self._boundary_edges[3:4, k]]

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

    def treat_robin(self, matrix, vector):
        # The number of bounadary edges
        num_boundary_edges = self._boundary_edges.shape[1]

        # Initialize the Robin vector `w` and matrix `r`
        w = np.zeros_like(vector)
        r = lil_matrix(matrix.shape)

        for k in range(num_boundary_edges):
            if self._boundary_edges[1, k] == -3:  # Robin boundary conditions
                # Get the element index associated withthe kth edge
                element_idx = self._boundary_edges[2, k]

                # Extract the global node coordinates to evaluate our integral on
                vertices = self._mesh.get_vertices(element_idx)

                # Extract the global node coordinates of the edge end points to evaluate our integral on
                edge_endpts = self._mesh.P[:, self._boundary_edges[3:4, k]]

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
