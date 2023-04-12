Getting Started
===============

This page details how to get started with the package `mlcvs`. 

Installation
------------
To install `mlcvs`, you will need an environment with the following packages:

* ``python > 3.7``
* ``numpy``
* ``pytorch >= 1.11``
* ``lightning``  

The following packages are optional, but they are recommended as they allow to use all of the helper functions contained in the utils module. 
* ``pandas`` (i/o)
* ``matplolib`` (plot)
* ``KDEpy`` or ``scikit-learn`` (compute free energy profiles via KDE)
* ``tqdm`` (monitor progress)

Once you have installed the requirements, you can install mlcvs by cloning the repository:
::

    git clone https://github.com/luigibonati/mlcvs.git 

and then installing it:

::

    cd mlcvs/
    pip install .

To install it in development (editable) mode:

::

    pip install -e .