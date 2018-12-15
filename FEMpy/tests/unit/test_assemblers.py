import numpy as np

from FEMpy import Mesh, FEBasis, Assemblers

mesh_1D_linear = Mesh.Interval1D(0, 1, 1/2, 'linear')
basis_1D_linear = FEBasis.IntervalBasis1D('linear')

mesh_1D_quadratic = Mesh.Interval1D(0, 1, 1/2, 'quadratic')
basis_1D_quadratic = FEBasis.IntervalBasis1D('quadratic')

mesh_2D_triangular_linear = Mesh.TriangularMesh2D(0, 1, 0, 1, 1/2, 1/2, 'linear')
basis_2D__triangular_linear = FEBasis.TriangularBasis2D('linear')


def coefficient_or_source_function(x):
    return 1



def test_matrix_assembly_1d_linear():
    matrix = Assemblers.assemble_matrix(coefficient_or_source_function, mesh_1D_linear,
                                        basis_1D_linear, basis_1D_linear,
                                        derivative_order_trial=1, derivative_order_test=1)
    assert np.allclose(matrix.toarray(), np.array([[2., -2., 0.],
                                                   [-2., 4., -2.],
                                                   [0., -2., 2.]]))


def test_matrix_assembly_1d_quadratic():
    matrix = Assemblers.assemble_matrix(coefficient_or_source_function, mesh_1D_quadratic,
                                        basis_1D_quadratic, basis_1D_quadratic,
                                        derivative_order_trial=1, derivative_order_test=1)
    assert np.allclose(matrix.toarray(), np.array([[4.6667, -5.3333, 0.6667, 0., 0.],
                                                   [-5.3333, 10.6667, -5.3333, 0., 0.],
                                                   [0.6667, -5.3333, 9.3333, -5.3333, 0.6667],
                                                   [0., 0., -5.3333, 10.6667, -5.3333],
                                                   [0., 0., 0.6667, -5.3333, 4.6667]]), rtol=1e-4, atol=1e-7)


def test_matrix_assembly_2d_linear():
    matrix = Assemblers.assemble_matrix(coefficient_or_source_function, mesh_2D_triangular_linear,
                                        basis_2D__triangular_linear, basis_2D__triangular_linear,
                                        derivative_order_trial=(1, 0), derivative_order_test=(1, 0))
    assert np.allclose(matrix.toarray(), np.array([[0.5, 0., 0., -0.5, 0., 0., 0., 0., 0.],
                                                   [0., 1., 0., 0., -1., 0., 0., 0., 0.],
                                                   [0., 0., 0.5, 0., 0., -0.5, 0., 0., 0.],
                                                   [-0.5, 0., 0., 1., 0., 0., -0.5, 0., 0.],
                                                   [0., -1., 0., 0., 2., 0., 0., -1., 0.],
                                                   [0., 0., -0.5, 0., 0., 1., 0., 0., -0.5],
                                                   [0., 0., 0., -0.5, 0., 0., 0.5, 0., 0.],
                                                   [0., 0., 0., 0., -1., 0., 0., 1., 0.],
                                                   [0., 0., 0., 0., 0., -0.5, 0., 0., 0.5]]))

# test_matrix_assembly_2d_quadratic omitted because the matrix is too large to type by hand.


def test_vector_assembly_1d_linear():
    vector = Assemblers.assemble_vector(coefficient_or_source_function, mesh_1D_linear,
                                        basis_1D_linear, derivative_order_test=0)
    assert np.allclose(vector, np.array([0.25, 0.5, 0.25]))


def test_vector_assembly_1d_quadratic():
    vector = Assemblers.assemble_vector(coefficient_or_source_function, mesh_1D_quadratic,
                                        basis_1D_quadratic, derivative_order_test=0)
    assert np.allclose(vector, np.array([0.0833, 0.3333, 0.1667, 0.3333, 0.0833]), rtol=1e-3, atol=1e-6)


def test_vector_assembly_2d_linear():
    vector = Assemblers.assemble_vector(coefficient_or_source_function, mesh_2D_triangular_linear,
                                        basis_2D__triangular_linear, derivative_order_test=(0,0))
    assert np.allclose(vector, np.array([0.0417, 0.1250, 0.0833, 0.1250, 0.25, 0.1250, 0.0833, 0.1250, 0.0417]),
                       rtol=1e-3, atol=1e-6)

# test_vector_assembly_2d_quadratic omitted because the vector is too large to type by hand.
