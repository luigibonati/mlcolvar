Contributing
============

We welcome contributions to the repository. 

- If you find a bug or want to ask for a new feature you can `open an issue <https://github.com/luigibonati/mlcolvar/issues>`_. 
- If you want to contribute new features (e.g. new methods for finding collective variables) or tutorials/examples, you can open a `pull request <https://github.com/luigibonati/mlcolvar/pulls>`_


Getting started
---------------

Create a fork of the repository and clone it. Then install the package in editable mode. 
In addition to the requirements you will also need additioanl packages for testing and building the documentation as detailed below.

Add a new feature
-----------------

Write a class/function, add tests, update doc etc...

Running tests
-------------

- we use pytest + nbmake extension for tutorials + codecov 

::
    pip XXX

test tests

::
    pytest -v --cov=mlcolvar --cov-report=xml mlcolvar/tests/

test notebooks

::
    pytest -v --nbmake docs/notebooks/ --ignore=docs/notebooks/tutorials/data/ --cov=mlcolvar --cov-append --cov-report=xml


Code formatting
---------------

we use black 

::
    pip install XXX


Contributing tutorials
----------------------

add them in docs/notebooks/tutorials

they will be tested by CI


Writing documentation
---------------------

you need the following additional packages:
- furo (design theme) , nbsphinx (display notebooks), sphinx-copybutton (copy button)

::
    pip install XXX


Write docstrings for classes and functions

add them to the docs. e.g. for a new cv add them in the relevant section of docs/api_cvs.rst, tutorials into docs/ttorials.rst

inside docs
::
    make html

the result will be in docs/_build