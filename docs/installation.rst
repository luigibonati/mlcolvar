Installation
============

The recommended way to install the package is using ``pip`` in a dedicated `virtual environment <installation.rst#create-a-virtual-environment>`_.

.. code-block:: bash

    # Activate here your Python virtual environment (e.g., with venv or conda).
    pip install mlcolvar


Download & Install from source
------------------------------

You can download the source code by cloning the repository locally using ``git``

.. code-block:: bash

    git clone https://github.com/luigibonati/mlcolvar.git

Alternatively, you can download a ``tar.gz`` or ``zip`` of the `latest release <https://github.com/luigibonati/mlcolvar/releases/latest>`_
or a specific release from the `releases page <https://github.com/luigibonati/mlcolvar/releases>`_.

To install `mlcolvar` from source, you will need an `environment <installation.rst#create-a-virtual-environment>`_
with the following **requirements**:

* ``python >= 3.8``
* ``numpy``
* ``pytorch >= 1.11``
* ``lightning > 1.18``

The following packages are optional requirements, but they are recommended as they allow to use all of the helper functions
contained in the utils module.

* ``pandas`` (i/o)
* ``matplolib`` (plot)
* ``KDEpy`` or ``scikit-learn`` (compute free energy profiles via KDE)
* ``tqdm`` (monitor training progress)

Finally, you can install the package by entering the downloaded (and unzipped) directory and executing

.. code-block:: bash

    # Activate here your Python virtual environment (e.g., with venv or conda).
    cd mlcolvar
    pip install .

If you are planning to `modify the code <contributing.rst>`_, we recommend you install in editable mode to have your
modifications automatically installed

.. code-block:: bash

    pip install -e .


Create a virtual environment
----------------------------

To create a virtual environment you can use either ``venv`` (which is supplied with Python 3) or if you prefer ``conda``.

With ``venv``, you can create a new virtual environment with

.. code-block:: bash

    python -m venv path/to/created/environment/folder

Then you can activate the environment to install packages in it.

.. code-block:: bash

    source path/to/created/environment/folder/bin/activate

Alternatively, if you are using ``conda`` you can create and activate the environment using

.. code-block:: bash

    conda create --name myenvname
    conda activate myenvname
