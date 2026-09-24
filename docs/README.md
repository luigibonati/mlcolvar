# Compiling mlcolvar's Documentation

The documentation for this project is built with [Sphinx](http://www.sphinx-doc.org/en/master/).

To compile the docs, install the documentation extras from the repository root:

```bash
python -m pip install -e ".[doc]"
```

Once installed, use the `Makefile` in this directory to build the HTML documentation:

```bash
make html
```

The compiled documentation will be available in the `_build` directory, typically under `_build/html/`, and can be viewed by opening `index.html`.

A configuration file for [Read the Docs](https://readthedocs.org/) (`readthedocs.yml`) is included at the top level of the repository. To host the documentation on Read the Docs, connect the repository to a Read the Docs project and configure the appropriate default branch.

Read the Docs installs the package with the `doc` extra as configured in `readthedocs.yml`.