.. _install:

Installation
============

At present, the only way to install FEMpy is via source. Installation with `pip <http://www.pip-installer.org/>`_ or
`conda <https://conda.io>`_ will come soon.

From Source
-----------

FEMpy is developed on `GitHub <https://github.com/floydie7/FEMpy>`_. To get the latest and cutting edge version, it is
easy to clone the source repository and install from there.

.. code-block:: bash

    git clone https://github.com/floydie7/FEMpy.git
    cd FEMpy
    python setup.py install

Test the Installation
---------------------

Unit and integration tests are provided with this package. To run the tests you will need `py.test <https://docs.pytest.org>`_. Simply run the
following command from the terminal in the package directory.

.. code-block:: bash

    pytest -v tests

