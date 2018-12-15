"""
Solvers.py
Author: Benjamin Floyd

Contains the Finite Element Method Solvers.
"""

import numpy as np
from scipy.integrate import quad
from scipy.sparse import linalg
from scipy.special import roots_legendre

from .Assemblers import assemble_matrix, assemble_vector
from .helpers import dbquad_triangle, copy_docstring_from


class Poisson1D(object):
    r"""
    Solves a one-dimensional Poisson equation.

    Uses finite element methods to solve a Poisson differential equation of the form

    .. math::
        - \frac{{\rm d}}{{\rm d} x}\left(c(x) \frac{{\rm d}}{{\rm d} x} u(x) \right) = f(x); a \leq x \leq b.

    with a combination of Dirichlet, Neumann, or Robin boundary conditions.

    .. warning:: Both end point boundary conditions cannot be Neumann as this may result in a loss of uniqueness of the
        solution.

    Parameters
    ----------
    mesh : :class:`FEMpy.Mesh.Interval1D`
        A :class:`Mesh` class defining the mesh and associated information matrices.
    fe_trial_basis, fe_test_basis : :class:`FEMpy.FEBasis.IntervalBasis1D`
        A :class:`FEBasis` class defining the finite element basis functions for the trial and test bases.
    boundary_conditions : :class:`FEMpy.Boundaries.BoundaryConditions`
        A :class:`BoundaryConditions` class defining the boundary conditions on the domain.

    Examples
    --------

    >>> import numpy as np
    >>> from FEMpy import Interval1D, IntervalBasis1D, BoundaryConditions, Poisson1D
    >>> mesh = Interval1D(0, 1, h=1/2, basis_type='linear')
    >>> basis = IntervalBasis1D('linear')
    >>> dirichlet_funct = lambda x: 0 if x == 0 else np.cos(1)
    >>> bcs = BoundaryConditions(mesh, ('dirichlet', 'dirichlet'), dirichlet_fun=dirichlet_funct)
    >>> coefficient_funct = lambda x: np.exp(x)
    >>> source_funct = lambda x: -np.exp(x) * (np.cos(x) - 2*np.sin(x) - x*np.cos(x) - x*np.sin(x))
    >>> poisson_eq = Poisson1D(mesh, basis, basis, bcs)
    >>> poisson_eq.solve(coefficient_funct, source_funct)
    array([0.        , 0.44814801, 0.54030231])

    """

    def __init__(self, mesh, fe_trial_basis, fe_test_basis, boundary_conditions):
        self._mesh = mesh
        self._fe_trial = fe_trial_basis
        self._fe_test = fe_test_basis
        self._boundary_conditions = boundary_conditions

        # This will be overwritten by the solver
        self._nodal_solution = None

    def solve(self, coeff_fun, source_fun):
        """
        Method that performs the finte element solution algorithm.

        Calls the assembly functions `FEMpy.Assemblers.assemble_matrix` and `FEMpy.Assemblers.assemble_vector` to create
        the stiffness matrix and load vector respectively. Then, applies the boundary condition treatments to the matrix
        and vector. Finally, solves the linear system :math:`A\mathbf{x} = \mathbf{b}.`

        Parameters
        ----------
        coeff_fun : function
            Function name of the coefficient function `c`(`x`) in the Poisson equation.
        source_fun : function
            The nonhomogeneous source function `f`(`x`) of the Poisson equation.

        Returns
        -------
        ndarray
            The nodal solution vector.
        """

        # Create our stiffness matrix
        stiffness_matrix = assemble_matrix(coeff_fun,
                                           mesh=self._mesh,
                                           trial_basis=self._fe_trial, test_basis=self._fe_test,
                                           derivative_order_trial=1, derivative_order_test=1)

        # Create our load vector
        load_vector = assemble_vector(source_fun,
                                      mesh=self._mesh,
                                      test_basis=self._fe_test,
                                      derivative_order_test=0)

        # Modify the stiffness matrix and the load vector to accommodate the boundary conditions specified.
        stiffness_matrix, load_vector = self._boundary_conditions.treat_robin(stiffness_matrix, load_vector)
        load_vector = self._boundary_conditions.treat_neumann(load_vector)
        stiffness_matrix, load_vector = self._boundary_conditions.treat_dirichlet(stiffness_matrix, load_vector)

        # Convert the stiffness matrix to a compressed sparse row matrix as it will be more efficient for matrix-vector
        # products
        stiffness_matrix = stiffness_matrix.tocsr()

        # Solve the system for our result
        self._nodal_solution = linalg.spsolve(stiffness_matrix, load_vector)

        return self._nodal_solution

    def fe_solution(self, x, local_sol, vertices, derivative_order):
        """
        Defines the functional solution piecewise on the finite on the finite elements.

        Uses the solution vector and the basis function to define a piecewise continuous solution over the element.

        Parameters
        ----------
        x : float or array_like
            A value or array of points to evaluate the function on.
        local_sol : array_like
            Finite element solution node vector local to the element `En`.
        vertices : array_like
            Global node coordinates for the mesh element `En`.
        derivative_order : int
            The derivative order to take the basis function to.

        Returns
        -------
        float
            Solution at all points in `x` in element.
        """

        # Set the basis type from the test basis
        basis_type = self._fe_test.basis_type

        if basis_type == 101:  # linear basis
            num_local_basis = 2
        elif basis_type == 102:  # quadratic basis
            num_local_basis = 3
        else:
            raise ValueError('Unknown basis type')

        fun_value = np.sum([local_sol[k] * self._fe_test.fe_local_basis(x, vertices, basis_idx=k,
                                                                         derivative_order=derivative_order)
                            for k in range(num_local_basis)], axis=0)

        return fun_value

    def l_inf_error(self, exact_sol):
        """
        Computes the L-infinity norm error.

        Computes the L-infinity norm error using the exact solution and the finite element function `fe_solution`.

        Parameters
        ----------
        exact_sol : function
            The analytical solution to compare the finite element solution against.

        Returns
        -------
        float
            The L-infinity norm error of the finite element solution over the domain evaluated element-wise.
        """

        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the element maximum error vector
        element_max = np.empty(num_elements)
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n)

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Generate grid of points local to the element
            node_points = roots_legendre(4)[0]
            element_points = (vertices[1] - vertices[0]) / 2 * node_points + (vertices[0] + vertices[1]) / 2

            # Compute the error on each evaluation node point in the element
            element_error = np.abs(exact_sol(element_points) - self.fe_solution(element_points, local_sol, vertices, 0))

            # Find the maximum error in the element
            element_max[n] = np.max(element_error)

        # Return the maximum error over all elements
        return np.max(element_max)

    def __l2_hsemi_norm_error(self, exact_sol, derivative_order):
        """
        Computes either the L2 norm error or the H1 semi-norm error.

        Computes either the L2 norm error or the H1 semi-norm error dependent on the derivative order specified . If
        ```derivative_order` == 1`` then the L2 norm error is computed, if ```derivative_order` == 1`` then the H1
        semi-norm is computed.

        .. note:: This is designed to be called via the `l2_error` and `h1_seminorm_error` methods which will provide
        the appropriate `derivative_order`.

        Parameters
        ----------
        exact_sol : function
            The analytical solution. If the H1 semi-norm error is desired, this must be the first derivative of the
            analytical solution.
        derivative_order : int
            The derivative order to take the basis function to.

        Returns
        -------
        float
            The L2 norm error or the H1 semi-norm error of the finite element solution over the domain evaluated
            element-wise.
        """

        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the elment error vector
        element_error = np.empty(num_elements)
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n)

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Define the integrand
            def integrand(x):
                return (exact_sol(x) - self.fe_solution(x, local_sol, vertices, derivative_order))**2

            # Integrate
            element_error[n] = quad(integrand, a=vertices[0], b=vertices[1])[0]

        # Return the sqrt of the sum of the errors
        return np.sqrt(np.sum(element_error))

    def l2_error(self, exact_sol):
        """
        The L2 norm error of the finite element solution compared against the given analytical solution.

        Parameters
        ----------
        exact_sol : function
            The analytical solution to the Poisson equation.

        Returns
        -------
        float
            The L2 norm error of the finite element solution over the domain evaluated element-wise.
        """

        return self.__l2_hsemi_norm_error(exact_sol, 0)

    def h1_seminorm_error(self, diff_exact_sol):
        """
        The H1 semi-norm error of the finite element solution compared against the given analyatical solution.

        Parameters
        ----------
        diff_exact_sol : function
            The first derivative of the analytical solution to the Poisson equation.

        Returns
        -------
        float
            The H1 semi-norm error of the finite element solution over the domain evaluated element-wise.
        """

        return self.__l2_hsemi_norm_error(diff_exact_sol, 1)


class Poisson2D(Poisson1D):
    r"""
    Solves a two-dimensional Poisson equation.

    Uses finite element methods to solve a Poisson differential equation of the form

    .. math::
        -\nabla\left(c(\mathbf{x}) \cdot \nabla u(\mathbf{x}) \right) = f(\mathbf{x}); \mathbf{x} \in \Omega

    with a combination of Dirichlet, Neumann, or Robin boundary conditions.

    .. warning:: All edge boundary conditions cannot be Neumann as this may result in a loss of uniqueness of the
        solution.

    Parameters
    ----------
    mesh : :class:`FEMpy.Mesh.TriangularMesh2D`
        A :class:`Mesh` class defining the mesh and associated information matrices.
    fe_trial_basis, fe_test_basis : :class:`FEMpy.FEBasis.IntervalBasis1D`
        A :class:`FEBasis` class defining the finite element basis functions for the trial and test bases.
    boundary_conditions : :class:`FEMpy.Boundaries.BoundaryConditions2D`
        A :class: `BoundaryConditions` class defining the boundary conditions on the domain.

    Examples
    --------

    >>> import numpy as np
    >>> from FEMpy import TriangularMesh2D, TriangularBasis2D, BoundaryConditions2D, Poisson2D
    >>> left, right, bottom, top = -1, 1, -1, 1
    >>> h = 1
    >>> def dirichlet_funct(coord):
    >>>     x, y = coord
    >>>     if x == -1:
    >>>         return np.exp(-1 + y)
    >>>     elif x == 1:
    >>>         return np.exp(1 + y)
    >>>     elif y == 1:
    >>>         return np.exp(x + 1)
    >>>     elif y == -1:
    >>>         return np.exp(x - 1)
    >>> coeff_funct = lambda coord: 1
    >>> source_funct = lambda coord: -2 * np.exp(coord[0] + coord[1])
    >>> mesh = TriangularMesh2D(left, right, bottom, top, h, h, 'linear')
    >>> basis = TriangularBasis2D('linear')
    >>> boundary_node_types = ['dirichlet'] * mesh.boundary_nodes.shape[1]
    >>> boundary_edge_types = ['dirichlet'] * (mesh.boundary_edges.shape[1]-1)
    >>> bcs = BoundaryConditions2D(mesh, boundary_node_types, boundary_edge_types, dirichlet_fun=dirichlet_funct)
    >>> poisson_eq = Poisson2D(mesh, basis, basis, bcs)
    >>> poisson_eq.solve(coeff_funct, source_funct)
    array([0.13533528, 0.36787944, 1., 0.36787944, 1.,  2.71828183, 1., 2.71828183, 7.3890561])

    """

    def __init__(self, mesh, fe_trial_basis, fe_test_basis, boundary_conditions):
        super().__init__(mesh, fe_trial_basis, fe_test_basis, boundary_conditions)

    def solve(self, coeff_fun, source_fun):
        """
        Method that performs the finte element solution algorithm.

        Calls the assembly functions `FEMpy.Assemblers.assemble_matrix` and `FEMpy.Assemblers.assemble_vector` to create
        the stiffness matrix and load vector respectively. Then, applies the boundary condition treatments to the matrix
        and vector. Finally, solves the linear system :math:`A\mathbf{x} = \mathbf{b}.`

        Parameters
        ----------
        coeff_fun : function
            Function name of the coefficient function `c`(`x`, `y`) in the Poisson equation.
        source_fun : function
            The nonhomogeneous source function `f`(`x`, `y`) of the Poisson equation.
        """

        # Create our stiffness matrix
        stiffness_matrix1 = assemble_matrix(coeff_fun,
                                            mesh=self._mesh,
                                            trial_basis=self._fe_trial, test_basis=self._fe_test,
                                            derivative_order_trial=(1, 0), derivative_order_test=(1, 0))
        stiffness_matrix2 = assemble_matrix(coeff_fun,
                                            mesh=self._mesh,
                                            trial_basis=self._fe_trial, test_basis=self._fe_test,
                                            derivative_order_trial=(0, 1), derivative_order_test=(0, 1))
        stiffness_matrix = (stiffness_matrix1.tocsr() + stiffness_matrix2.tocsr()).tolil()

        # Create our load vector
        load_vector = assemble_vector(source_fun,
                                      mesh=self._mesh,
                                      test_basis=self._fe_test,
                                      derivative_order_test=(0, 0))

        # Modify the stiffness matrix and the load vector to accommodate the boundary conditions specified.
        stiffness_matrix, load_vector = self._boundary_conditions.treat_robin(stiffness_matrix, load_vector)
        load_vector = self._boundary_conditions.treat_neumann(load_vector)
        stiffness_matrix, load_vector = self._boundary_conditions.treat_dirichlet(stiffness_matrix, load_vector)

        # Convert the stiffness matrix to a compressed sparse row matrix as it will be more efficient for matrix-vector
        # products
        stiffness_matrix = stiffness_matrix.tocsr()

        # Solve the system for our result
        self._nodal_solution = linalg.spsolve(stiffness_matrix, load_vector)

        return self._nodal_solution

    def fe_solution(self, coords, local_sol, vertices, derivative_order):
        """
        Defines the functional solution piecewise on the finite on the finite elements.

        Uses the solution vector and the basis function to define a piecewise continuous solution over the element.

        Parameters
        ----------
        coords : float or array_like
            A value or array of points to evaluate the function on.
        local_sol : array_like
            Finite element solution node vector local to the element `En`.
        vertices : array_like
            Global node coordinates for the mesh element `En`.
        derivative_order : tuple of int
            The derivative orders in the x- and y-directions to take the basis function to.

        Returns
        -------
        float
            Solution at all points in `coords` in element.
        """

        # Set the basis type from the test basis
        basis_type = self._fe_test.basis_type

        if basis_type == 201:  # linear basis
            num_local_basis = 3
        elif basis_type == 202:  # quadratic basis
            num_local_basis = 6
        else:
            raise ValueError('Unknown basis type')

        fun_value = np.sum([local_sol[k] * self._fe_test.fe_local_basis(coords, vertices, basis_idx=k,
                                                                        derivative_order=derivative_order)
                            for k in range(num_local_basis)])

        return fun_value

    @copy_docstring_from(Poisson1D.l_inf_error)
    def l_inf_error(self, exact_sol):
        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the element maximum error vector
        element_max = np.empty(num_elements)
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n).T

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Generate grid of points local to the element
            # grid_points = sample_points_in_triangle(50)
            grid_points = np.array([[(1+0)/2, (1-0)*(1+0)/4],
                                    [(1+np.sqrt(3/5))/2, (1-np.sqrt(3/5))*(1+np.sqrt(3/5))/4],
                                    [(1+np.sqrt(3/5))/2, (1-np.sqrt(3/5))*(1-np.sqrt(3/5))/4],
                                    [(1-np.sqrt(3/5))/2, (1+np.sqrt(3/5))*(1+np.sqrt(3/5))/4],
                                    [(1-np.sqrt(3/5))/2, (1+np.sqrt(3/5))*(1-np.sqrt(3/5))/4],
                                    [(1+0)/2, (1-0)*(1+np.sqrt(3/5))/4],
                                    [(1+0)/2, (1-0)*(1-np.sqrt(3/5))/4],
                                    [(1+np.sqrt(3/5))/2, (1-np.sqrt(3/5))*(1+0)/4],
                                    [(1-np.sqrt(3/5))/2, (1+np.sqrt(3/5))*(1+0)/4]]).T

            # Transform the sampled grid point into our triangle
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]

            # The affine transformation from the standard triangle to our element triangle is
            new_x = (-x1 + x2) * grid_points[0] + (-x1 + x3) * grid_points[1] + x1
            new_y = (-y1 + y2) * grid_points[0] + (-y1 + y3) * grid_points[1] + y1

            element_points = np.vstack([new_x, new_y])

            # Compute the error on each evaluation node point in the element
            element_error = np.abs(exact_sol(element_points) - self.fe_solution(element_points, local_sol, vertices,
                                                                                derivative_order=(0, 0)))

            # Find the maximum error in the element
            element_max[n] = np.max(element_error)

        # Return the maximum error over all elements
        return np.max(element_max)

    def __l2_hsemi_norm_error(self, exact_sol, derivative_order):
        """
        Computes either the L2 norm error or the H1 semi-norm error.

        Computes either the L2 norm error or the H1 semi-norm error dependent on the derivative order specified . If
        ```derivative_order` == 1`` then the L2 norm error is computed, if ```derivative_order` == 1`` then the H1
        semi-norm is computed.

        .. note:: This is designed to be called via the `l2_error` and `h1_seminorm_error` methods which will provide
        the appropriate `derivative_order`.

        Parameters
        ----------
        exact_sol : function
            The analytical solution. If the H1 semi-norm error is desired, this must be the first derivative of the
            analytical solution.
        derivative_order : tuple of int
            The derivative orders in the x- and y-directions to take the basis function to.

        Returns
        -------
        float
            The L2 norm error or the H1 semi-norm error of the finite element solution over the domain evaluated
            element-wise.
        """
        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the elment error vector
        element_error = np.empty(num_elements)
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n).T

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Define the integrand
            def integrand(x):
                return (exact_sol(x) - self.fe_solution(x, local_sol, vertices, derivative_order))**2

            # Integrate
            element_error[n] = dbquad_triangle(integrand, vertices)[0]

        # Return the sqrt of the sum of the errors
        return np.sqrt(np.sum(element_error))

    @copy_docstring_from(Poisson1D.l2_error)
    def l2_error(self, exact_sol):
        return self.__l2_hsemi_norm_error(exact_sol, (0, 0))

    def h1_seminorm_error(self, diff_exact_sol):
        """
        The H1 semi-norm error of the finite element solution compared against the given analyatical solution.

        Parameters
        ----------
        diff_exact_sol : tuple of function
            A tuple of first derivatives in the x- and the y- directions the analytical solution to the Poisson equation
            respectively.

        Returns
        -------
        float
            The full H1 semi-norm error of the finite element solution over the domain evaluated element-wise.
        """

        dx_exact_sol, dy_exact_sol = diff_exact_sol

        h1_x_error = self.__l2_hsemi_norm_error(dx_exact_sol, (1, 0))
        h1_y_error = self.__l2_hsemi_norm_error(dy_exact_sol, (0, 1))

        return np.sqrt(h1_x_error**2 + h1_y_error**2)
