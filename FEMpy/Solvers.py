"""
Solvers.py
Author: Benjamin Floyd

Contains the Finite Element Method Solvers.
"""

import numpy as np
from scipy import linalg
from scipy.integrate import quad

from .Assemblers import assemble_matrix, assemble_vector
from .helpers import dbquad_triangle


class Poisson1D(object):
    def __init__(self, mesh, fe_trial_basis, fe_test_basis, boundary_conditions):
        self._mesh = mesh
        self._fe_trial = fe_trial_basis
        self._fe_test = fe_test_basis
        self._boundary_conditions = boundary_conditions

        # This will be overwritten by the solver
        self._nodal_solution = None

    def solve(self, coeff_fun, source_fun):
        # Generate the P and T mesh node information matrices
        self._mesh.generate_p_t_matrices()

        # Generate the Pb and Tb finite element node information matrices
        self._mesh.generate_pb_tb_matrices()

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

        # Solve the system for our result
        self._nodal_solution = linalg.solve(stiffness_matrix, load_vector)

    def fe_solution(self, x, local_sol, vertices, derivative_order):
        # Set the basis type from the test basis
        basis_type = self._fe_test.basis_type

        if basis_type == 101:  # linear basis
            num_local_basis = 2
        elif basis_type == 102:  # quadratic basis
            num_local_basis = 3
        else:
            raise ValueError('Unknown basis type')

        fun_value = np.sum([local_sol[k] * self._fe_test(x, vertices, basis_idx=k, derivative_order=derivative_order)
                            for k in range(num_local_basis)])

        return fun_value

    def l_inf_error(self, exact_sol):
        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the element maximum error vector
        element_max = np.empty((1, num_elements))
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n)

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Generate grid of points local to the element
            element_points = np.linspace(vertices[0], vertices[1])

            # Compute the error on each evaluation node point in the element
            element_error = np.abs(exact_sol(element_points) - self.fe_solution(element_points, local_sol, vertices, 0))

            # Find the maximum error in the element
            element_max[n] = np.max(element_error)

        # Return the maximum error over all elements
        return np.max(element_max)

    def __l2_hsemi_norm_error(self, exact_sol, derivative_order):
        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the elment error vector
        element_error = np.empty((1, num_elements))
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
        return self.__l2_hsemi_norm_error(exact_sol, 0)

    def h1_seminorm_error(self, diff_exact_sol):
        return self.__l2_hsemi_norm_error(diff_exact_sol, 1)


class Poisson2D(Poisson1D):
    def __init__(self, mesh, fe_trial_basis, fe_test_basis, boundary_conditions):
        super().__init__(mesh, fe_trial_basis, fe_test_basis, boundary_conditions)

    def solve(self, coeff_fun, source_fun):
        # Generate the P and T mesh node information matrices
        self._mesh.generate_p_t_matrices()

        # Generate the Pb and Tb finite element node information matrices
        self._mesh.generate_pb_tb_matrices()

        # Create our stiffness matrix
        stiffness_matrix1 = assemble_matrix(coeff_fun,
                                            mesh=self._mesh,
                                            trial_basis=self._fe_trial, test_basis=self._fe_test,
                                            derivative_order_trial=(1, 0), derivative_order_test=(1, 0))
        stiffness_matrix2 = assemble_matrix(coeff_fun,
                                            mesh=self._mesh,
                                            trial_basis=self._fe_trial, test_basis=self._fe_test,
                                            derivative_order_trial=(0, 1), derivative_order_test=(0, 1))
        stiffness_matrix = stiffness_matrix1 + stiffness_matrix2

        # Create our load vector
        load_vector = assemble_vector(source_fun,
                                      mesh=self._mesh,
                                      test_basis=self._fe_test,
                                      derivative_order_test=(0, 0))

        # Modify the stiffness matrix and the load vector to accommodate the boundary conditions specified.
        stiffness_matrix, load_vector = self._boundary_conditions.treat_robin(stiffness_matrix, load_vector)
        load_vector = self._boundary_conditions.treat_neumann(load_vector)
        stiffness_matrix, load_vector = self._boundary_conditions.treat_dirichlet(stiffness_matrix, load_vector)

        # Solve the system for our result
        self._nodal_solution = linalg.solve(stiffness_matrix, load_vector)

    def fe_solution(self, coords, local_sol, vertices, derivative_order):
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

    def l_inf_error(self, exact_sol):
        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the element maximum error vector
        element_max = np.empty((1, num_elements))
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n)

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Generate grid of points local to the element
            element_points = np.linspace(vertices[0], vertices[1])  # Todo: these need to be local nodes

            # Compute the error on each evaluation node point in the element
            element_error = np.abs(exact_sol(element_points) - self.fe_solution(element_points, local_sol, vertices,
                                                                                derivative_order=(0, 0)))

            # Find the maximum error in the element
            element_max[n] = np.max(element_error)

        # Return the maximum error over all elements
        return np.max(element_max)

    def __l2_hsemi_norm_error(self, exact_sol, derivative_order):
        # Get the number of elements from the mesh
        num_elements = self._mesh.num_elements_x

        # Initialize the elment error vector
        element_error = np.empty((1, num_elements))
        for n in range(num_elements):
            # Extract the global node coordinates for the element E_n
            vertices = self._mesh.get_vertices(n)

            # Select for the solution at the local finite element nodes
            local_sol = self._nodal_solution[self._mesh.Tb[:, n]]

            # Define the integrand
            def integrand(x):
                return (exact_sol(x) - self.fe_solution(x, local_sol, vertices, derivative_order))**2

            # Integrate
            element_error[n] = dbquad_triangle(integrand, vertices)[0]

        # Return the sqrt of the sum of the errors
        return np.sqrt(np.sum(element_error))

    def l2_error(self, exact_sol):
        return self.__l2_hsemi_norm_error(exact_sol, (0, 0))

    def h1_seminorm_error(self, diff_exact_sol):
        dx_exact_sol, dy_exact_sol = diff_exact_sol

        h1_x_error = self.__l2_hsemi_norm_error(dx_exact_sol, (1, 0))
        h1_y_error = self.__l2_hsemi_norm_error(dy_exact_sol, (0, 1))

        return np.sqrt(h1_x_error**2 + h1_y_error**2)
