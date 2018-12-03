"""
FEMpy : A Python finite element solver.
Author: Benjamin Floyd
"""

# Import the relevant modules so they are conviently accessible on import.
from .Boundaries import *
from .FEBasis import *
from .Mesh import *
from .Solvers import *

# If the user imports the entire package only provide them the front facing packages
__all__ = [s for s in dir() if not s.startswith('_')]
