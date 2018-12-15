import numpy as np
import scipy.io as sio

import FEMpy

# Load matlab linear solution vector files
matlab_save_linear = sio.loadmat('poison_2d_linear_h0125_solution_vector.mat')
matlab_linear_solution_vector = matlab_save_linear['solution'].T
matlab_save_quadratic = sio.loadmat('poison_2d_quadratic_h0125_solution_vector.mat')
matlab_quadratic_solution_vector = matlab_save_quadratic['solution'].T


def coefficient_function(coords):
    return 1


def source_function(coord):
    x, y = coord
    return -2 * np.exp(x + y)


def dirichlet_function(coord):
    x, y = coord
    if x == -1:
        return np.exp(-1 + y)
    elif x == 1:
        return np.exp(1 + y)
    elif y == 1:
        return np.exp(x + 1)
    elif y == -1:
        return np.exp(x - 1)


def neumann_function(coord):
    x, y = coord
    if y == -1:
        return -np.exp(x - 1)


def analytic_solution(coord):
    x, y = coord
    return np.exp(x + y)


def dx_analytic_solution(coord):
    x, y = coord
    return np.exp(x + y)


def dy_analytic_solution(coord):
    x, y = coord
    return np.exp(x + y)


class TestLinearElements(object):
    def setup(self):
        self.mesh = FEMpy.TriangularMesh2D(-1, 1, -1, 1, 1/8, 1/8, 'linear')
        self.basis = FEMpy.TriangularBasis2D('linear')
        boundary_node_types = ['dirichlet', *['neumann']*15, *['dirichlet']*17, *['dirichlet']*16, *['dirichlet']*15]
        boundary_edge_types = [*['neumann']*16, *['dirichlet']*16, *['dirichlet']*16, *['dirichlet']*16]
        self.bcs = FEMpy.BoundaryConditions2D(self.mesh,
                                              boundary_node_types=boundary_node_types,
                                              boundary_edge_types=boundary_edge_types,
                                              test_basis=self.basis,
                                              dirichlet_fun=dirichlet_function,
                                              neumann_fun=neumann_function,
                                              coeff_fun=coefficient_function)
        self.poisson_eq = FEMpy.Poisson2D(self.mesh, self.basis, self.basis, self.bcs)

    def test_solution_vector(self):
        nodal_solution_vector = self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        assert np.allclose(nodal_solution_vector,matlab_linear_solution_vector)

    def test_l_infinity_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_infinity_norm_error = self.poisson_eq.l_inf_error(analytic_solution)
        assert np.abs(l_infinity_norm_error - 1.3358e-02) <= 1e-5

    def test_l_2_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_2_norm_error = self.poisson_eq.l2_error(analytic_solution)
        assert np.abs(l_2_norm_error - 5.1224e-03) <= 1e-5

    def test_h_1_seminorm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        h_1_seminorm_error = self.poisson_eq.h1_seminorm_error((dx_analytic_solution, dy_analytic_solution))
        assert np.abs(h_1_seminorm_error - 1.8523e-01) <= 1e-5


class TestQuadraticElements(object):
    def setup(self):
        self.mesh = FEMpy.TriangularMesh2D(-1, 1, -1, 1, 1 / 8, 1 / 8, 'quadratic')
        self.basis = FEMpy.TriangularBasis2D('quadratic')
        boundary_node_types = ['dirichlet', *['neumann'] * 30, *['dirichlet'] * 34 , *['dirichlet'] * 32,
                               *['dirichlet'] * 30]
        boundary_edge_types = [*['neumann'] * 16, *['dirichlet'] * 16, *['dirichlet'] * 16, *['dirichlet'] * 16]
        self.bcs = FEMpy.BoundaryConditions2D(self.mesh,
                                              boundary_node_types=boundary_node_types,
                                              boundary_edge_types=boundary_edge_types,
                                              test_basis=self.basis,
                                              dirichlet_fun=dirichlet_function,
                                              neumann_fun=neumann_function,
                                              coeff_fun=coefficient_function)
        self.poisson_eq = FEMpy.Poisson2D(self.mesh, self.basis, self.basis, self.bcs)

    def test_solution_vector(self):
        nodal_solution_vector = self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        assert np.allclose(nodal_solution_vector, matlab_quadratic_solution_vector)

    def test_l_infinity_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_infinity_norm_error = self.poisson_eq.l_inf_error(analytic_solution)
        assert np.abs(l_infinity_norm_error - 1.0956e-04) <= 1e-5

    def test_l_2_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_2_norm_error = self.poisson_eq.l2_error(analytic_solution)
        assert np.abs(l_2_norm_error - 3.9285e-05) <= 1e-5

    def test_h_1_seminorm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        h_1_seminorm_error = self.poisson_eq.h1_seminorm_error((dx_analytic_solution, dy_analytic_solution))
        assert np.abs(h_1_seminorm_error - 2.9874e-03) <= 1e-5