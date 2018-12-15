import numpy as np

from FEMpy import Assemblers, Boundaries, FEBasis, Mesh


# Stand in for the various functions needed
def generic_function(x):
    return 1


mesh_1D = Mesh.Interval1D(0, 1, 1/2, 'linear')
basis_1D = FEBasis.IntervalBasis1D('linear')
matrix_1D = Assemblers.assemble_matrix(generic_function, mesh_1D, basis_1D, basis_1D, 1, 1)
vector_1D = Assemblers.assemble_vector(generic_function, mesh_1D, basis_1D, 0)

mesh_2D = Mesh.TriangularMesh2D(0, 1, 0, 1, 1/2, 1/2, 'linear')
basis_2D = FEBasis.TriangularBasis2D('linear')
matrix_2D = Assemblers.assemble_matrix(generic_function, mesh_2D, basis_2D, basis_2D, (1, 0), (1,0))
vector_2D = Assemblers.assemble_vector(generic_function, mesh_2D, basis_2D, (0, 0))

boundary_condition_1D_dirichlet_neumann = Boundaries.BoundaryConditions(mesh_1D,
                                                                        boundary_types=['dirichlet', 'neumann'],
                                                                        dirichlet_fun=generic_function,
                                                                        neumann_fun=generic_function,
                                                                        coeff_fun=generic_function)

boundary_condition_1D_dirichlet_robin = Boundaries.BoundaryConditions(mesh_1D,
                                                                      boundary_types=['dirichlet', 'robin'],
                                                                      dirichlet_fun=generic_function,
                                                                      robin_fun_p=generic_function,
                                                                      robin_fun_q=generic_function,
                                                                      coeff_fun=generic_function)

boundary_condition_1D_neumann_robin = Boundaries.BoundaryConditions(mesh_1D,
                                                                    boundary_types=['neumann', 'robin'],
                                                                    neumann_fun=generic_function,
                                                                    robin_fun_p=generic_function,
                                                                    robin_fun_q=generic_function,
                                                                    coeff_fun=generic_function)

boundary_condition_2D_all_types = Boundaries.BoundaryConditions2D(mesh_2D,
                                                                  boundary_node_types=['dirichlet', 'dirichlet', 'dirichlet', 'neumann', 'neumann', 'neumann', 'neumann', 'robin'],
                                                                  boundary_edge_types=['dirichlet', 'dirichlet', 'neumann', 'neumann', 'neumann', 'robin', 'robin'],
                                                                  trial_basis=basis_2D,
                                                                  test_basis=basis_2D,
                                                                  dirichlet_fun=generic_function,
                                                                  neumann_fun=generic_function,
                                                                  robin_fun_p=generic_function,
                                                                  robin_fun_q=generic_function,
                                                                  coeff_fun=generic_function)


class TestBoundaryNodeMatrix1D(object):
    def test_dirichlet_neumann_matrix(self):
        boundary_node_matrix_1d = boundary_condition_1D_dirichlet_neumann._boundary_nodes
        assert np.allclose(boundary_node_matrix_1d, np.array([[-1, -2], [0, 2]]))

    def test_dirichlet_robin_matrix(self):
        boundary_node_matrix_1d = boundary_condition_1D_dirichlet_robin._boundary_nodes
        assert np.allclose(boundary_node_matrix_1d, np.array([[-1, -3], [0, 2]]))

    def test_neumann_robin_matrix(self):
        boundary_node_matrix_1d = boundary_condition_1D_neumann_robin._boundary_nodes
        assert np.allclose(boundary_node_matrix_1d, np.array([[-2, -3], [0, 2]]))


class TestBoundaryMatrices2D(object):
    def test_2d_boundary_node_matrix(self):
        boundary_node_matrix_2d = boundary_condition_2D_all_types._boundary_nodes
        assert np.allclose(boundary_node_matrix_2d, np.array([[-1, -1, -1, -2, -2, -2, -2, -3],
                                                              [0, 3, 6, 7, 8, 5, 2, 1]]))

    def test_2d_boundary_edge_matrix(self):
        boundary_edge_matrix_2d = boundary_condition_2D_all_types._boundary_edges
        assert np.allclose(boundary_edge_matrix_2d, np.array([[-1, -1, -2, -2, -2, -3, -3],
                                                              [0, 4, 5, 7, 7, 3, 2],
                                                              [0, 3, 6, 7, 8, 5, 2],
                                                              [3, 6, 7, 8, 5, 2, 1]]))


class TestBoundaryConditionTreatments1D(object):
    def test_dirichlet_treatment_matrix_1d(self):
        treated_matrix, _ = boundary_condition_1D_dirichlet_neumann.treat_dirichlet(matrix_1D, vector_1D)
        assert np.allclose(treated_matrix.toarray(), np.array([[1., 0., 0.],
                                                               [-2., 4., -2.],
                                                               [0., -2., 2.]]))

    def test_dirichlet_treatment_vector_1d(self):
        _, treated_vector = boundary_condition_1D_dirichlet_neumann.treat_dirichlet(matrix_1D, vector_1D)
        assert np.allclose(treated_vector, np.array([1., 0.5, 0.25]))

    def test_neumann_treatment_vector_1d(self):
        treated_vector = boundary_condition_1D_dirichlet_neumann.treat_neumann(vector_1D)
        assert np.allclose(treated_vector, np.array([0.25, 0.5, 1.25]))

    def test_robin_treatment_matrix_1d(self):
        treated_matrix, _ = boundary_condition_1D_dirichlet_robin.treat_robin(matrix_1D, vector_1D)
        assert np.allclose(treated_matrix.toarray(), np.array([[2., -2., 0.],
                                                               [-2., 4., -2.],
                                                               [0., -2., 3.]]))

    def test_robin_treatment_vector_1d(self):
        _, treated_vector = boundary_condition_1D_dirichlet_robin.treat_robin(matrix_1D, vector_1D)
        assert np.allclose(treated_vector, np.array([0.25, 0.5, 1.25]))


class TestBoundaryConditionTreatment2D(object):
    def test_dirichlet_treatment_matrix_2d(self):
        treated_matrix, _ = boundary_condition_2D_all_types.treat_dirichlet(matrix_2D, vector_2D)
        assert np.allclose(treated_matrix.toarray(), np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                               [0., 1., 0., 0., -1., 0., 0., 0., 0.],
                                                               [0., 0., 0.5, 0., 0., -0.5, 0., 0., 0.,],
                                                               [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                                               [0., -1., 0., 0., 2., 0., 0., -1., 0.],
                                                               [0., 0., -0.5, 0., 0., 1., 0., 0., -0.5],
                                                               [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                                               [0., 0., 0., 0., -1., 0., 0., 1., 0.],
                                                               [0., 0., 0., 0., 0., -0.5, 0., 0., 0.5]]))

    def test_dirichlet_treatment_vector_2d(self):
        _, treated_vector = boundary_condition_2D_all_types.treat_dirichlet(matrix_2D, vector_2D)
        assert np.allclose(treated_vector, np.array([1., 0.125, 0.0833, 1., 0.25, 0.125, 1., 0.125, 0.0417]),
                           rtol=1e-3, atol=1e-6)

    def test_neumann_treatment_vector_2d(self):
        treated_vector = boundary_condition_2D_all_types.treat_neumann(vector_2D)
        assert np.allclose(treated_vector, np.array([0.0417, 0.125, 0.0833, 0.125, 0.25, 0.375, 0.3333, 0.625,0.5417]),
                           rtol=1e-3, atol=1e-6)

    def test_robin_treatment_matrix_2d(self):
        treated_matrix, _ = boundary_condition_2D_all_types.treat_robin(matrix_2D, vector_2D)
        assert np.allclose(treated_matrix.toarray(), np.array([[0.5, 0., 0., -0.5, 0., 0., 0., 0., 0.],
                                                               [0., 1.1667, 0.0833, 0., -1., 0., 0., 0., 0.],
                                                               [0., 0.0833, 0.8333, 0., 0., -0.4167, 0., 0., 0.],
                                                               [-0.5, 0., 0., 1., 0., 0., -0.5, 0., 0.],
                                                               [0., -1., 0., 0., 2., 0., 0., -1., 0.],
                                                               [0., 0., -0.4167, 0., 0., 1.1667, 0., 0., -0.5],
                                                               [0., 0., 0., -0.5, 0., 0., 0.5, 0., 0.],
                                                               [0., 0., 0., 0., -1., 0., 0., 1., 0.],
                                                               [0., 0., 0., 0., 0., -0.5, 0., 0., 0.5]]),
                           rtol=1e-3, atol=1e-6)

    def test_robin_treatment_vector_2d(self):
        _, treated_vector = boundary_condition_2D_all_types.treat_robin(matrix_2D, vector_2D)
        assert np.allclose(treated_vector, np.array([0.0417, 0.375, 0.5833, 0.125, 0.25, 0.375, 0.0833, 0.125, 0.0417]),
                           rtol=1e-3, atol=1e-6)
