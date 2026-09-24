Contributing
============

We very much welcome contributions to the repository! If you want to contribute bugfixes, new features (e.g., new methods for finding collective variables), or documentation (e.g., tutorials and examples), this guide is for you.

Getting started
---------------

Before starting to work on your changes, we suggest you `open an issue <https://github.com/luigibonati/mlcolvar/issues>`_ (or comment on a relevant existing one) describing very briefly the changes you would like to implement. The developers might be able to give you additional guidance based on the long-term plans for the library and point you to the easiest implementation path from the start.

Then:

1. `Create a fork <https://help.github.com/articles/fork-a-repo>`_ of this repository on GitHub.
2. `Clone <https://help.github.com/articles/cloning-a-repository>`_ your fork of the repository on your local machine.
3. Install the package locally, preferably in a virtual environment (see :doc:`installation`), from the cloned source in editable mode, together with the dependencies required for testing and building the documentation:

   .. code-block:: bash

      # Activate your Python virtual environment (e.g., with venv or conda).
      cd mlcolvar
      pip install -e ".[doc,test]"

Once your environment is set up, you are ready to implement your changes.

Overview of the GitHub workflow
-------------------------------

Regardless of the type of contribution, the workflow on GitHub is the same.

1. Implement and test your changes (see below for guidelines specific for :ref:`code <contributing-bugfixes-and-new-features>`, :ref:`documentation <contributing-documentation>`, and :ref:`tutorials <contributing-tutorials>`).
2. When you are ready to receive feedback on your changes, navigate to your fork of ``mlcolvar`` on GitHub and `open a pull request <https://help.github.com/articles/using-pull-requests>`_ (PR). Note that after you launch the PR, all subsequent commits will be automatically added to the open PR and tested.
3. When you are ready to be considered for merging, check the "Ready to go" box on the PR page to let the mlcolvar developers know that the changes are complete.
4. A developer will review your changes and may suggest modifications.
5. Once a developer marks the PR as "approved" for merging and the continuous integration tests pass, the PR will be merged into the main codebase.

.. _contributing-bugfixes-and-new-features:

Contributing bugfixes and new features
--------------------------------------

* If you are implementing a new CV, the documentation has guides on how to implement one `from scratch <https://mlcolvar.readthedocs.io/en/latest/notebooks/tutorials/adv_newcv_scratch.html>`_ or by `subclassing an existing one <https://mlcolvar.readthedocs.io/en/latest/notebooks/tutorials/adv_newcv_subclass.html>`_.
* Stick to the :ref:`coding style guidelines <coding-style-guidelines>` when possible.
* :ref:`Add tests <writing-tests>` for your new code! If you are contributing a bugfix, chances are our current test suite does not cover this case, and a test should be written to avoid future regressions. If you are contributing a new feature, your tests should make sure it is working as expected.
* If you are writing a new feature or changing the behavior of the library, :ref:`add or modify the docstrings <contributing-documentation>` describing the behavior of your code.

.. _contributing-documentation:

Contributing documentation
--------------------------

The main documentation of ``mlcolvar`` is inside the ``docs/`` folder. It is written using the `reStructuredText markup syntax <https://docutils.sourceforge.io/rst.html>`_ and automatically built in HTML format using `Sphinx <https://www.sphinx-doc.org/>`_ and published on `Read the Docs <https://mlcolvar.readthedocs.io/en/latest/>`_.

Classes and functions should be documented in the Python code using `NumPy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_. Sphinx will take care of collecting the docstrings in the code and compiling the API documentation.

Writing short working examples of code usage in docstrings is usually tremendously helpful and very much appreciated. In NumPy-style docstrings, these are written in the `Examples section <https://numpydoc.readthedocs.io/en/latest/format.html#examples>`_.

Moreover, if the example is written as a Python `doctest <https://docs.python.org/3/library/doctest.html>`_ (roughly, by starting each line of code in the example with ``>>>``), it can be executed automatically to help ensure that the example does not become outdated. To make sure your doctests run smoothly, add the following at the bottom of the ``myfile.py`` file containing the docstring:

.. code-block:: python

   if __name__ == "__main__":
       import doctest
       doctest.testmod()

Then simply run:

.. code-block:: bash

   python myfile.py

When modifying the documentation, especially when using reStructuredText syntax, it is a good idea to build the documentation locally and check the result. To install only the documentation dependencies from the repository root, run:

.. code-block:: bash

   pip install ".[doc]"

Then build the documentation with:

.. code-block:: bash

   cd docs
   make html

The resulting HTML pages will be built in ``docs/_build/html/`` and can be viewed with any browser.

.. _contributing-tutorials:

Contributing tutorials
----------------------

* Tutorials are available in ``mlcolvar`` as Jupyter notebooks saved in ``docs/notebooks/tutorials/``.
* As for the library's code, stick to the :ref:`coding style guidelines <coding-style-guidelines>` when possible.
* Make sure the notebook runs from start to finish before opening the PR, as it will be automatically tested using ``pytest``'s `nbmake <https://github.com/treebeardtech/nbmake>`_ plugin.

.. _writing-tests:

Writing tests
-------------

``mlcolvar`` uses `pytest <https://docs.pytest.org/>`_ for automatic testing. We highly recommend running the tests locally before submitting a PR. You can install the test dependencies from the repository root with:

.. code-block:: bash

   pip install ".[test]"

If you are writing tests for code in the file ``mlcolvar/example/folder/file.py``, then your tests should be implemented as functions whose names start with ``test_`` and placed in ``mlcolvar/tests/test_example_folder_file.py``.

You can run the entire test suite with:

.. code-block:: bash

   pytest mlcolvar/tests/

``pytest`` will automatically discover all test functions. To run the tests in a single file, use:

.. code-block:: bash

   pytest mlcolvar/tests/test_my_file.py

To run a single test function within a file, use:

.. code-block:: bash

   pytest mlcolvar/tests/test_my_file.py::test_my_function

**Pro tip** - Consider using the ``@pytest.mark.parametrize`` decorator (see the `pytest documentation <https://docs.pytest.org/en/stable/how-to/parametrize.html>`_) to automate testing multiple test cases, and ``pytest.raises`` (see the `pytest documentation on expected exceptions <https://docs.pytest.org/en/stable/how-to/assert.html#assertions-about-expected-exceptions>`_) to test error handling.

.. _coding-style-guidelines:

Coding style guidelines
-----------------------

Using coding style guidelines makes it much easier to read, understand, and search through the code. ``mlcolvar`` adheres to Python's `PEP 8 convention <https://peps.python.org/pep-0008/>`_.

If you are unfamiliar with PEP 8, you might like using a formatter such as `Black <https://black.readthedocs.io/en/stable/>`_. You can install it with:

.. code-block:: bash

   pip install black

If you want to format Jupyter notebooks, install it with:

.. code-block:: bash

   pip install "black[jupyter]"

Then run ``black`` on the file you are editing:

.. code-block:: bash

   black your_file