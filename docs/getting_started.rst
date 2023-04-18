Getting Started
===============

This page details how to get started with the package `mlcolvar`. 

Installation
------------
To install `mlcolvar`, you will need an environment with the following packages:

* ``python > 3.7``
* ``numpy``
* ``pytorch >= 1.11``
* ``lightning``  

The following packages are optional, but they are recommended as they allow to use all of the helper functions contained in the utils module. 
* ``pandas`` (i/o)
* ``matplolib`` (plot)
* ``KDEpy`` or ``scikit-learn`` (compute free energy profiles via KDE)
* ``tqdm`` (monitor progress)

Once you have installed the requirements, you can install mlcolvar by cloning the repository:
::

    git clone https://github.com/luigibonati/mlcolvar.git 

and then installing it:

::

    cd mlcolvar/
    pip install .

To install it in development (editable) mode:

::

    pip install -e .