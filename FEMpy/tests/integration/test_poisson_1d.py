import numpy as np

import FEMpy


def coefficient_function(x):
    return np.exp(x)


def source_function(x):
    return -np.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))


def dirichlet_function(x):
    if x == 0:
        return 0


def neumann_function(x):
    if x == 1:
        return np.cos(1) - np.sin(1)


def analytical_sol(x):
    return x * np.cos(x)


def dx_analytical_sol(x):
    return np.cos(x) - x * np.sin(x)


class TestLinearElements(object):
    def setup(self):
        self.mesh = FEMpy.Interval1D(0, 1, 1/4, 'linear')
        self.basis = FEMpy.IntervalBasis1D('linear')
        self.bcs = FEMpy.BoundaryConditions(self.mesh,
                                            ['dirichlet', 'neumann'],
                                            dirichlet_fun=dirichlet_function,
                                            neumann_fun=neumann_function,
                                            coeff_fun=coefficient_function)
        self.poisson_eq = FEMpy.Poisson1D(self.mesh, self.basis, self.basis, self.bcs)

    def test_solution_vector(self):
        nodal_solution_vector = self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        assert np.allclose(nodal_solution_vector, np.array([2.9317e-16, 0.24174, 0.43690, 0.54469, 0.53351]))

    def test_l_infinity_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_infinity_norm_error = self.poisson_eq.l_inf_error(analytical_sol)
        assert np.abs(l_infinity_norm_error - 2.0464e-02) <= 1e-5

    def test_l_2_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_2_norm_error = self.poisson_eq.l2_error(analytical_sol)
        assert np.abs(l_2_norm_error - 1.1205e-02) <= 1e-5

    def test_h_1_seminorm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        h_1_seminorm_error = self.poisson_eq.h1_seminorm_error(dx_analytical_sol)
        assert np.abs(h_1_seminorm_error - 1.0542e-01) <= 1e-5


class TestQuadraticElements(object):
    def setup(self):
        self.mesh = FEMpy.Interval1D(0, 1, 1 / 4, 'quadratic')
        self.basis = FEMpy.IntervalBasis1D('quadratic')
        self.bcs = FEMpy.BoundaryConditions(self.mesh,
                                            ['dirichlet', 'neumann'],
                                            dirichlet_fun=dirichlet_function,
                                            neumann_fun=neumann_function,
                                            coeff_fun=coefficient_function)
        self.poisson_eq = FEMpy.Poisson1D(self.mesh, self.basis, self.basis, self.bcs)

    def test_solution_vector(self):
        nodal_solution_vector = self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        assert np.allclose(nodal_solution_vector, np.array([-1.3260e-15, 0.12407,  0.24223, 0.34899, 0.43880, 0.50689, 0.54878, 0.56090, 0.54031]),
                           rtol=1e-4, atol=1e-7)

    def test_l_infinity_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_infinity_norm_error = self.poisson_eq.l_inf_error(analytical_sol)
        assert np.abs(l_infinity_norm_error - 3.3279e-04) <= 1e-5

    def test_l_2_norm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        l_2_norm_error = self.poisson_eq.l2_error(analytical_sol)
        assert np.abs(l_2_norm_error - 2.1050e-04) <= 1e-5

    def test_h_1_seminorm_error(self):
        self.poisson_eq.solve(coeff_fun=coefficient_function, source_fun=source_function)
        h_1_seminorm_error = self.poisson_eq.h1_seminorm_error(dx_analytical_sol)
        assert np.abs(h_1_seminorm_error - 5.4213e-03) <= 1e-5
