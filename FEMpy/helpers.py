"""
helpers.py
Author: Benjamin Floyd

Contains helper functions.
"""

import numpy as np
from scipy.integrate import quad, dblquad


def det_jacobian(u, v, vertices):
    """Determinant of Jacobian matrix."""

    # Unpack our vertex coordinates
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # Derivatives
    dxdu = (1 - v) * x2 + v * x3 - x1
    dxdv = u * x3 - u * x2
    dydu = (1 - v) * y2 + v * y3 - y1
    dydv = u * y3 - u * y2

    return np.abs(dxdu * dydv - dxdv * dydu)


def dbquad_triangle(integrand, vertices, **kwargs):
    """

    Parameters
    ----------
    integrand
    vertices
    kwargs

    Returns
    -------

    """

    # Unpack our vertex coordinates
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # Transformation to the unit square
    def unit_sq_trans(u, v, c1, c2, c3):
        return (1 - u) * c1 + u * ((1 - v) * c2 + v * c3)

    # Transformation for the integrand
    def integrand_trans(u, v):
        # Transform x and y to the unit square
        x = unit_sq_trans(u, v, x1, x2, x3)
        y = unit_sq_trans(u, v, y1, y2, y3)

        coords = (x, y)

        # Redefine our integrand
        integral_value = integrand(coords, vertices, **kwargs) * det_jacobian(u, v, vertices)

        return integral_value

    # Perform the double integration using quadrature in the transformed space
    integral, error = dblquad(integrand_trans, 0, 1, lambda x: 0, lambda x: 1, epsrel=1e-6, epsabs=0)

    return integral, error


def line_integral(integrand, endpoints, **kwargs):
    # Extract our end point coordinates
    x0 = endpoints[1, 1]
    y0 = endpoints[2, 1]
    x1 = endpoints[1, 2]
    y1 = endpoints[2, 2]

    # Parameterize our curve
    def x(t):
        return x0 + (x1 - x0) * t

    def y(t):
        return y0 + (y1 - y0) * t

    # Compute the magnitude of `||r'(t)||`
    r_prime_mag = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    # Change our integrand to be in terms of our parametrization
    def param_integrand(t):
        return integrand((x(t), y(t)), **kwargs) * r_prime_mag

    # Preform our integration
    int_value = quad(param_integrand, a=0, b=1)[0]

    return int_value
