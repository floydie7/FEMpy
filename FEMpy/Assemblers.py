"""
Assemblers.py
Author: Benjamin Floyd

Contains the matrix and vector assembler methods.
"""

from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.sparse import lil_matrix

from .helpers import dbquad_triangle


def assemble_matrix(coeff_funct, mesh, trial_basis, test_basis, derivative_order_trial, derivative_order_test):
    """
    Construct the finite element stiffness matrix.

    Parameters
    ----------
    coeff_funct : function
        Function name of the coefficient function `c`(`x`(,`y`,...)) in a Poisson equation.
    mesh : {:class: FEMpy.Mesh.Interval1D, :class: FEMpy.Mesh.TriangularMesh2D}
        A :class:`Mesh` class defining the mesh and associated information matrices.
    trial_basis, test_basis : {:class: FEMpy.FEBasis.IntervalBasis1D, :class: FEMpy.FEBasis.TriangularBasis2D}
        A :class: `FEBasis` class defining the finite element basis functions for the trial and test bases.
    derivative_order_trial, derivative_order_test : int or tuple of int
        The derivative order to be applied to the finite element basis functions. If basis function is one-dimensional,
        this should be specified as an int. Otherwise, the derivative order should be specified as a tuple with the
        orders corresponding to the coordinate axes in the basis functions. e.g., ``(`x_order`, `y_order`,...)``.

    Returns
    -------
    sparse matrix
        The finite element stiffness matrix as a Compressed Sparse Row matrix.
    """

    # Determine the size of the matrix
    num_equations, num_local_trial = _basis_type_parser(trial_basis.basis_type, mesh)
    num_unknowns, num_local_test = _basis_type_parser(test_basis.basis_type, mesh)

    # Initialize the matrix as a sparse, row-based linked list matrix.
    A = lil_matrix((num_equations, num_unknowns))

    for n in range(mesh.T.shape[1]):
        # Generate the vertices for each element
        vertices = mesh.get_vertices(n)

        for alpha, beta in product(range(num_local_trial), range(num_local_test)):
            # Set up our integrand
            def integrand(coords):
                return (coeff_funct(coords)
                        * trial_basis.fe_local_basis(coords, vertices, basis_idx=alpha,
                                                     derivative_order=derivative_order_trial)
                        * test_basis.fe_local_basis(coords, vertices, basis_idx=beta,
                                                    derivative_order=derivative_order_test))

            # Integrate using adaptive Gaussian quadrature
            if test_basis.basis_type in [101, 102]:
                int_value = quad(integrand, a=vertices[0], b=vertices[1])[0]

            elif test_basis.basis_type in [201, 202]:
                int_value, _ = dbquad_triangle(integrand, vertices, basis_idx=(alpha, beta),
                                               derivative_order=(derivative_order_trial, derivative_order_test))
            else:
                raise ValueError('Unknown basis type')

            A[mesh.Tb[beta, n], mesh.Tb[alpha, n]] += int_value

    # Return our matrix as a compressed sparse row matrix as it will be more efficient for matrix vector products
    return A.tocsr()


def assemble_vector(source_funct, mesh, test_basis, derivative_order_test):
    """
    Constructs the finite element load vector.

    Parameters
    ----------
    source_funct : function
        The nonhomogeneous source function `f`(`x`(,`y`,...)) of the Poisson equation.
    mesh : {:class: FEMpy.Mesh.Interval1D, :class: FEMpy.Mesh.TriangularMesh2D}
        A :class:`Mesh` class defining the mesh and associated information matrices.
    test_basis : {:class: FEMpy.FEBasis.IntervalBasis1D, :class: FEMpy.FEBasis.TriangularBasis2D}
        A :class: `FEBasis` class defining the finite element basis functions for the test basis.
    derivative_order_test : int or tuple of int
        The derivative order to be applied to the finite element basis function. If basis function is one-dimensional,
        this should be specified as an int. Otherwise, the derivative order should be specified as a tuple with the
        orders corresponding to the coordinate axes in the basis functions. e.g., ``(`x_order`, `y_order`,...)``.

    Returns
    -------
    ndarray
        The finite element load vector.
    """

    # Determine the size of the vector
    num_unknowns, num_local_test = _basis_type_parser(test_basis.basis_type, mesh)

    # Initialize the vector
    b = np.zeros(int(num_unknowns))

    for n in range(mesh.T.shape[1]):
        # Extract the global node coordinates to evaluate our integral on
        vertices = mesh.get_vertices(n)

        for beta in range(num_local_test):
            # Set up our integrand
            def integrand(coords, *args):
                return (source_funct(coords)
                        * test_basis.fe_local_basis(coords, vertices=args[0], basis_idx=args[1],
                                                    derivative_order=args[2]))

            # Integrate using adaptive Gaussian quadrature
            if test_basis.basis_type in [101, 102]:
                int_value = quad(integrand, a=vertices[0], b=vertices[1],
                                 args=(vertices, beta, derivative_order_test))[0]
            elif test_basis.basis_type in [201, 202]:
                int_value, _ = dbquad_triangle(integrand, vertices, basis_idx=beta,
                                               derivative_order=derivative_order_test)
            else:
                raise ValueError('Unknown basis type')

            b[mesh.Tb[beta, n]] += int_value

    return b


def _basis_type_parser(basis_type, mesh):
    """
    Parses the basis type and mesh information.

    Uses the `basis_type` and the `mesh` information to determine the number of unknowns or equations for our linear
    system as well as the number of local basis functions in each element.

    Parameters
    ----------
    basis_type : int
        A integer code identifying the basis type.
    mesh : :class: `FEMpy.Mesh`
        A :class: `Mesh` class defining the mesh and corresponding information matrices.

    Returns
    -------
    num_unknowns_eqs : int
        Number of unknown variables or equations in linear system.
    num_local_basis_fns : int
        Number of local basis functions in finite element.
    """
    if basis_type == 101:
        # Pull the number of elements from the mesh information
        num_elements_x = mesh.num_elements_x

        # Determine the number of unknown variables and basis functions needed
        num_unknowns_eqs = num_elements_x + 1
        num_local_basis_fns = 2

    elif basis_type == 102:
        # Pull the number of elements from the mesh information
        num_elements_x = mesh.num_elements_x

        # Determine the number of unknown variables and basis functions needed
        num_unknowns_eqs = 2 * num_elements_x + 1
        num_local_basis_fns = 3

    elif basis_type == 201:
        # Pull the number of elements from the mesh information
        num_elements_x = mesh.num_elements_x
        num_elements_y = mesh.num_elements_y

        # Determine the number of unknown variables and basis functions needed
        num_unknowns_eqs = (num_elements_x + 1) * (num_elements_y + 1)
        num_local_basis_fns = 3

    elif basis_type == 202:
        # Pull the number of elements from the mesh information
        num_elements_x = mesh.num_elements_x
        num_elements_y = mesh.num_elements_y

        # Determine the number of unknown variables and basis functions needed
        num_unknowns_eqs = (2 * num_elements_x + 1) * (2 * num_elements_y + 1)
        num_local_basis_fns = 6

    else:
        raise ValueError('Unknown basis type.')

    return num_unknowns_eqs, num_local_basis_fns
