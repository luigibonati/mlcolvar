Installation
============

The recommended way to install ``mlcolvar`` is using ``pip`` in a dedicated
virtual environment.

``mlcolvar`` requires Python 3.11 or later.

.. code-block:: bash

    # Activate your Python virtual environment (e.g., with venv or conda).
    pip install mlcolvar


Download and install from source
--------------------------------

You can download the source code by cloning the repository locally using
``git``:

.. code-block:: bash

    git clone https://github.com/luigibonati/mlcolvar.git

Alternatively, you can download a ``tar.gz`` or ``zip`` archive of the
`latest release <https://github.com/luigibonati/mlcolvar/releases/latest>`_
or a specific release from the
`releases page <https://github.com/luigibonati/mlcolvar/releases>`_.

To install ``mlcolvar`` from source, enter the downloaded repository and run:

.. code-block:: bash

    # Activate your Python virtual environment (e.g., with venv or conda).
    cd mlcolvar
    pip install .

Runtime dependencies are installed automatically by ``pip``. The authoritative
list of dependencies is defined in ``pyproject.toml``.

If you are planning to `modify the code <contributing.rst>`_, we recommend
installing the package in editable mode so that local modifications are
immediately available:

.. code-block:: bash

    pip install -e .

To check that the library is working properly, install the test dependencies
and run the test suite:

.. code-block:: bash

    pip install "mlcolvar[test]"
    pytest --pyargs mlcolvar.tests


Create a virtual environment
----------------------------

To create a virtual environment, you can use either ``venv`` (included with
Python) or ``conda``.

With ``venv``, create a new environment using a Python 3.11 or later
interpreter:

.. code-block:: bash

    python -m venv path/to/created/environment/folder

Then activate the environment before installing the package.

On Linux or macOS:

.. code-block:: bash

    source path/to/created/environment/folder/bin/activate

On Windows:

.. code-block:: powershell

    path\to\created\environment\folder\Scripts\activate

Alternatively, with ``conda``:

.. code-block:: bash

    conda create --name myenvname python=3.11
    conda activate myenvname