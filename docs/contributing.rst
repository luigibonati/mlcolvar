Contributing
============

We very much welcome contributions to the repository! If you want to contribute bugfixes, new features (e.g., new methods
for finding collective variables), or documentation (e.g., tutorials and examples), this guide is for you.


Getting started
---------------

Before starting to work on your changes, we suggest you `open an issue <https://github.com/luigibonati/mlcolvar/issues>`_
(or comment on a relevant existing one) describing very briefly the changes you would like to implement. The developers
might be able to give you additional guidance based on the long-term plans for the library and point you to the easiest
implementation path from the start. Then

1. `Create a fork <https://help.github.com/articles/fork-a-repo>`_ of this repository on GitHub.
2. `Clone <https://help.github.com/articles/cloning-a-repository>`_ your fork of the repository on your local machine.
3. Install the package locally (preferably in a `virtual environment <installation.rst#create-a-virtual-environment>`_
   from the cloned source in editable mode so that your changes will be automatically installed.

   .. code-block:: bash

      # Activate here your Python virtual environment (e.g., with venv or conda).
      cd mlcolvar
      pip install -e .

4. In order to perform the regtests and build the documentation you need to install additional packages:

    .. code-block:: bash

      pip install mlcolvar[docs,test]

Once your environment is set up you are ready to implement your changes.


Overview of the GitHub workflow
-------------------------------

Regardless of the type of contribution, the workflow on GitHub is the same.

1. Implement and test your changes (see below for guidelines specific for `code <contributing.rst#Contributing-bugfixes-and-new-features>`_,
   `documentation <contributing.rst#Contributing-documentation>`_, and `tutorials <contributing.rst#Contributing-tutorials>`_).
2. When you are ready to receive feedback on your changes, navigate to your fork of ``mlcolvar`` on GitHub and
   `open a pull request <https://help.github.com/articles/using-pull-requests>`_ (PR). Note that after you launch the PR, all
   subsequent commits will be automatically added to the open PR and tested.
3. When you're ready to be considered for merging, check the "Ready to go" box on the PR page to let the mlcolvar devs
   know that the changes are complete.
4. A developer will review your changes and eventually make suggestions for modifications.
5. Once a developer mark the PR as "approved" for merging and the continuous integration tests pass, the PR will be merged
   in the main codebase.


Contributing bugfixes and new features
--------------------------------------

* If you are implementing a new CV, the documentation have guides on how to implement one `from scratch <https://mlcolvar.readthedocs.io/en/latest/notebooks/tutorials/adv_newcv_scratch.html>`_
  or by `subclassing an existing one <https://mlcolvar.readthedocs.io/en/latest/notebooks/tutorials/adv_newcv_subclass.html>`_.
* Stick to the `coding style guidelines <contributing.rst#Coding-style-guidelines>`_ when possible.
* `Add tests <contributing.rst#Writing-tests>`_ for your new code! If are contributing a bugfix, chances are our current test suite
  does not cover this case, adn a test should be written to avoid future regressions. If you are contributing a new feature,
  your tests should make sure it is working as expected.
* If you are writing a new features or changing the behavior of the library, `add/modify the docstrings <contributing.rst#Contributing-documentation>`_
  describing the behavior of your code.


Contributing documentation
--------------------------

The main documentation of ``mlcolvar`` is inside the ``docs/`` folder. It is written using using the `reStructuredText markup syntax <https://docutils.sourceforge.io/rst.html>`_
and automatically built in html format using `Sphinx <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/index.html>`_ and
pushed on `readthedocs.io <https://mlcolvar.readthedocs.io/en/latest/>`_.

Classes and functions should be documented in the Python code using `numpy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
Sphinx will take care of collecting the docstrings in the code and compiling a documentation of the library's API.

Writing short working examples of code usage in the docstring is usually tremendously helpful and very much appreciated. In numpy-style
docstrings, these are written in the `Examples section <https://numpydoc.readthedocs.io/en/latest/format.html#examples>`_.
Moreover, if the example is written as a Python `doctest <https://docs.python.org/3/library/doctest.html>`_ (roughly, just
start each line of code in your example with ``>>>``), this will be automatically run by the continuous integration, ensuring
that the example will not go out-of-date in future code changes. To make sure your doctest run smoothly, add at the bottom
of the ``myfile.py`` file including the docstring

.. code-block:: python

    if __name__ == '__main__':
        import doctest
        doctest.testmod()

and simply run

.. code-block:: python

    python myfile.py

Finally, when modifying the documentation (especially when using reStructuredText syntax), it is a very good idea to build the
documentation to check the result. To do this you will need to install these additional packages:

.. code-block:: bash

    pip install furo nbsphinx sphinx-copybutton

or more simply using:

.. code-block:: bash

    pip install mlcolvar[doc]

Then, you can build the docs via the command

.. code-block:: bash

    cd docs/
    make html

the resulting ``html`` pages will be built in ``docs/_build/`` and can be visualized with any browser.


Contributing tutorials
----------------------

* Tutorials are available in ``mlcolvar`` in the form of Jupyter notebooks saved in ``docs/notebooks/tutorials/``.
* As for the library's code, stick to the `coding style guidelines <contributing.rst#Coding-style-guidelines>`_ when possible.
* Make sure the notebook runs from start to end before opening the PR as it will be automatically tested using ``pytest``'s
  `nbmake <https://github.com/treebeardtech/nbmake>`_ plugin.


Writing tests
-------------

``mlcolvar`` uses `pytest <https://docs.pytest.org/en/7.3.x/>`_ for automatic testing. We highly recommend installing
``pytest`` and run your tests locally before submitting the PR. You can install pytest with

.. code-block:: bash

      pip install pytest

If you are writing tests for code in the file ``mlcolvar/example/folder/file.py``, then your tests should be implemented
as functions whose name start with ``test_``, and they should be placed in ``mlcolvar/tests/test_example_folder_file.py``.
You can run the entire test suite with the command

.. code-block:: bash

    pytest mlcolvar/tests/

and ``pytest`` will automatically discover all the test functions. If you want to run the tests in a single file, use

.. code-block:: bash

    pytest mlcolvar/tests/test_my_file.py

or a single function within a file

.. code-block:: bash

    pytest mlcolvar/tests/test_my_file.py::test_my_function

**Pro tip** - Consider using the ``@pytest.mark.parametrize`` decorator (see `docs <https://docs.pytest.org/en/7.1.x/how-to/parametrize.html>`_)
to automatize testing multiple test cases and ``pytest.raises`` (see `docs <https://docs.pytest.org/en/7.1.x/how-to/assert.html#assertions-about-expected-exceptions>`_)
to test error handling.


Coding style guidelines
-----------------------

Using coding style guidelines makes it much easier to read, understand, and search through the code. ``mlcolvar`` adheres
to Python's `PEP8 convention <https://peps.python.org/pep-0008>`_.

If you are unfamiliar with PEP8, you might like using a linter for automatic formatting. A popular one is `black <https://black.readthedocs.io/en/stable/>`_.
You can install it through pip

.. code-block:: bash

    pip install black

If you want to format Jupyter notebooks, install it with the command

.. code-block:: bash

    pip install black[jupyter]

Then run ``black`` on the file you are editing.

.. code-block:: bash

    black your_file
