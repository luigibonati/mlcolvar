Getting Started
===============

This page details how to get started with mlcvs. 

Installation
------------
To install mlcvs, you will need an environment with the following packages:

* ``Python > 3.7``
* ``NumPy``
* ``Pandas``
* ``PyTorch``
* ``Matplotlib`` (plotting functions)
* ``Scikit-learn`` (FES estimation via KDE)

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