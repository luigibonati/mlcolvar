Installation
============

It is recommended to install the package and its dependencies in a virtual environment. 

Requirements
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

Download
--------

You can download the 'latest <https://github.com/luigibonati/mlcolvar/releases/latest>'_ release (or a specific one) from the 'releases <https://github.com/luigibonati/mlcolvar/releases>'_ page.

In alternative, you can clone the Github repository:
::

    git clone https://github.com/luigibonati/mlcolvar.git 


Install
-------

Once you have downloaded the package you can install it by entering the directory and executing: 
::

    cd mlcolvar/
    pip install .

