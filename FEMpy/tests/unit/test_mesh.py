import numpy as np

from FEMpy import Mesh


class TestLinear1D(object):
    def setup(self):
        self.mesh = Mesh.Interval1D(0, 1, 1/2, 'linear')

    def test_p_matrix(self):
        p_matrix = self.mesh.P
        assert np.allclose(p_matrix, np.array([0., 0.5, 1.]))

    def test_t_matrix(self):
        t_matrix = self.mesh.T
        assert np.array_equal(t_matrix, np.array([[0, 1], [1, 2]]))

    def test_pb_matrix(self):
        pb_matrix = self.mesh.Pb
        assert np.allclose(pb_matrix, np.array([0., 0.5, 1.]))

    def test_tb_matrix(self):
        tb_matrix = self.mesh.Tb
        assert np.array_equal(tb_matrix, np.array([[0, 1], [1, 2]]))

    def test_boundary_node_coordinates(self):
        boundary_node_coordinates = self.mesh.boundary_nodes
        assert np.allclose(boundary_node_coordinates, [0., 1.])

    def test_number_of_elements(self):
        number_of_elements_x_direction = self.mesh.num_elements_x
        assert np.abs(number_of_elements_x_direction - 2.) <= 1e-4

    def test_get_vertices(self):
        element_0_vertices = self.mesh.get_vertices(0)
        assert np.allclose(element_0_vertices, [0., 0.5])


class TestQuadratic1D(object):
    def setup(self):
        self.mesh = Mesh.Interval1D(0, 1, 1/2, 'quadratic')

    def test_p_matrix(self):
        p_matrix = self.mesh.P
        assert np.allclose(p_matrix, np.array([0., 0.5, 1.]))

    def test_t_matrix(self):
        t_matrix = self.mesh.T
        assert np.array_equal(t_matrix, np.array([[0, 1], [1, 2]]))

    def test_pb_matrix(self):
        pb_matrix = self.mesh.Pb
        assert np.allclose(pb_matrix, np.array([0., 0.25, 0.5, 0.75, 1.]))

    def test_tb_matrix(self):
        tb_matrix = self.mesh.Tb
        assert np.array_equal(tb_matrix, np.array([[0, 2], [2, 4], [1, 3]]))

    def test_boundary_node_coordinates(self):
        boundary_node_coordinates = self.mesh.boundary_nodes
        assert np.allclose(boundary_node_coordinates, [0., 1.])

    def test_number_of_elements(self):
        number_of_elements_x_direction = self.mesh.num_elements_x
        assert np.abs(number_of_elements_x_direction - 2.) <= 1e-4

    def test_get_vertices(self):
        element_0_vertices = self.mesh.get_vertices(0)
        assert np.allclose(element_0_vertices, [0., 0.5])


class TestLinear2DTriangular(object):
    def setup(self):
        self.mesh = Mesh.TriangularMesh2D(0, 1, 0, 1, 1/2, 1/2, 'linear')

    def test_p_matrix(self):
        p_matrix = self.mesh.P
        assert np.allclose(p_matrix, np.array([[0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.],
                                               [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]]))

    def test_t_matrix(self):
        t_matrix = self.mesh.T
        assert np.array_equal(t_matrix, np.array([[0, 1, 1, 2, 3, 4, 4, 5],
                                                  [3, 3, 4, 4, 6, 6, 7, 7],
                                                  [1, 4, 2, 5, 4, 7, 5, 8]]))

    def test_pb_matrix(self):
        pb_matrix = self.mesh.Pb
        assert np.allclose(pb_matrix, np.array([[0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.],
                                               [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]]))

    def test_tb_matrix(self):
        tb_matrix = self.mesh.Tb
        assert np.array_equal(tb_matrix, np.array([[0, 1, 1, 2, 3, 4, 4, 5],
                                                  [3, 3, 4, 4, 6, 6, 7, 7],
                                                  [1, 4, 2, 5, 4, 7, 5, 8]]))

    def test_boundary_node_coordinates(self):
        boundary_node_coordinates = self.mesh.boundary_nodes
        assert np.allclose(boundary_node_coordinates, np.array([[0., 0.5, 1., 1., 1., 0.5, 0., 0],
                                                                [0., 0., 0., 0.5, 1., 1., 1., 0.5]]))

    def test_number_of_elements_x_direction(self):
        number_of_elements_x_direction = self.mesh.num_elements_x
        assert np.abs(number_of_elements_x_direction - 2.) <= 1e-4

    def test_number_of_elemnents_y_direction(self):
        number_of_elements_y_direction = self.mesh.num_elements_y
        assert np.abs(number_of_elements_y_direction - 2.) <= 1e-4

    def test_get_vertices(self):
        element_0_vertices = self.mesh.get_vertices(0)
        assert np.allclose(element_0_vertices, np.array([[0., 0.5, 0.],
                                                         [0., 0., 0.5]]))


class TestQuadratic2DTriangular(object):
    def setup(self):
        self.mesh = Mesh.TriangularMesh2D(0, 1, 0, 1, 1/2, 1/2, 'quadratic')

    def test_p_matrix(self):
        p_matrix = self.mesh.P
        assert np.allclose(p_matrix, np.array([[0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.],
                                               [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]]))

    def test_t_matrix(self):
        t_matrix = self.mesh.T
        assert np.array_equal(t_matrix, np.array([[0, 1, 1, 2, 3, 4, 4, 5],
                                                  [3, 3, 4, 4, 6, 6, 7, 7],
                                                  [1, 4, 2, 5, 4, 7, 5, 8]]))

    def test_pb_matrix(self):
        pb_matrix = self.mesh.Pb
        assert np.allclose(pb_matrix, np.array([[0., 0., 0., 0., 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 1., 1., 1., 1., 1.],
                                               [0., 0.25, 0.5, 0.75, 1., 0., 0.25, 0.5, 0.75, 1., 0., 0.25, 0.5, 0.75, 1., 0., 0.25, 0.5, 0.75, 1., 0., 0.25, 0.5, 0.75, 1.]]))

    def test_tb_matrix(self):
        tb_matrix = self.mesh.Tb
        assert np.array_equal(tb_matrix, np.array([[0, 2, 2, 4, 10, 12, 12, 14],
                                                   [10, 10, 12, 12, 20, 20, 22, 22],
                                                   [2, 12, 4, 14, 12, 22, 14, 24],
                                                   [5, 6, 7, 8, 15, 16, 17, 18],
                                                   [6, 11, 8, 13, 16, 21, 18, 23],
                                                   [1, 7, 3, 9, 11, 17, 13, 19]]))

    def test_boundary_node_coordinates(self):
        boundary_node_coordinates = self.mesh.boundary_nodes
        assert np.allclose(boundary_node_coordinates, np.array([[0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0.],
                                                                [0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 0.75, 0.5, 0.25]]))

    def test_number_of_elements_x_direction(self):
        number_of_elements_x_direction = self.mesh.num_elements_x
        assert np.abs(number_of_elements_x_direction - 2.) <= 1e-4

    def test_number_of_elemnents_y_direction(self):
        number_of_elements_y_direction = self.mesh.num_elements_y
        assert np.abs(number_of_elements_y_direction - 2.) <= 1e-4

    def test_get_vertices(self):
        element_0_vertices = self.mesh.get_vertices(0)
        assert np.allclose(element_0_vertices, np.array([[0., 0.5, 0.],
                                                         [0., 0., 0.5]]))
