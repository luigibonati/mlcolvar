Installation
============

This page details how to get started with the package `mlcolvar`. 

The library is based on the PyTorch machine learning library, and the high-level Lightning package. 
The latter to simplifies the overall model training workflow and allows us focusing only on the CV design and optimization. 
Although it can be used as a stand-alone tool (e.g., for analysis of MD simulations), the main purpose is to create variables that can be used in combination with enhanced sampling methods through PLUMED C++ software. Hence we will need to deploy the optimized model in a python-independent format in order to use it during the MD simulations. 

Installation
------------
To install `mlcolvar`, you will need an environment with the following packages:

* ``python > 3.7``
* ``numpy``
* ``pytorch >= 1.11``
* ``lightning > 1.18``  

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