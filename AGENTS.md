# AGENTS.md

This file gives coding agents repository-wide guidance for working on `mlcolvar`.

## Project Purpose

`mlcolvar` is a Python library for building machine-learned collective variables (CVs) for enhanced-sampling molecular simulations. It is built on PyTorch and Lightning. Users typically prepare tensor or molecular-graph data, construct a CV model, train it with a Lightning `Trainer`, and export the trained model for inference or use with PLUMED.

Keep scientific behavior, tensor shapes, autograd, device placement, and model export compatibility in mind when changing public code. Do not silently detach tensors, move data to CPU, or change numerical precision unless the API explicitly requires it.

## Repository Layout

- `mlcolvar/core/` contains reusable numerical and neural-network building blocks.
  - `estimators/` implements methods such as PCA, TICA, LDA, and generator estimators.
  - `loss/` contains reusable loss functions.
  - `nn/` contains feed-forward and graph neural-network architectures.
  - `transform/` contains differentiable preprocessing, descriptors, and other tensor transforms.
- `mlcolvar/cvs/` contains trainable collective-variable models grouped by learning approach: supervised, unsupervised, time-lagged, committor, generator, and multitask.
- `mlcolvar/cvs/cv.py` defines `BaseCV`, the central Lightning module. CVs compose ordered model blocks through `DEFAULT_BLOCKS`, `MODEL_BLOCKS`, and `BLOCKS`.
- `mlcolvar/data/` contains dictionary-based datasets, Lightning data modules, loaders, time-lagged data helpers, and graph data structures.
- `mlcolvar/io/` handles COLVAR and graph-related input/output.
- `mlcolvar/explain/` contains sensitivity and sparse-model interpretation tools.
- `mlcolvar/utils/` contains training, plotting, free-energy, export, and data utilities.
- `mlcolvar/cli/` implements the `mlcolvar-fes` and `mlcolvar-deltag` entry points.
- `mlcolvar/tests/` contains the pytest suite and its test data. Tests are distributed as part of the package.
- `docs/` contains Sphinx documentation and executable tutorial/example notebooks.
- `plumed_interfaces/` contains integration material for deploying models with PLUMED.

## Development Setup

Work from the repository root. Use an isolated Python environment.

```bash
python -m pip install -U pip
python -m pip install -e ".[test]"
```

For documentation work, install both test and documentation extras:

```bash
python -m pip install -e ".[test,doc]"
```

The package declares Python 3.8 or newer. CI currently exercises Python 3.11 through 3.13 on Linux, Windows, and macOS, so avoid platform-specific assumptions and preserve the declared minimum version unless the project metadata is deliberately changed.

## Making Changes

- Keep changes focused and follow existing neighboring implementations before introducing new abstractions.
- Preserve public APIs unless a breaking change is explicitly requested. Update package `__init__.py` exports when adding a public class or function.
- New CVs should normally subclass `BaseCV`, declare their block layout, and implement the appropriate training behavior. Respect the distinction between list-defined feed-forward models and supplied `FeedForward` or `BaseGNN` instances.
- New transforms should subclass `Transform`, implement `forward`, and use `setup_from_datamodule` only when parameters depend on training data.
- Keep reusable math and model components in `core`; keep task-level Lightning models in `cvs`.
- Preserve tensor batch dimensions and clearly document expected input/output shapes. Cover graph and feed-forward paths separately when both are affected.
- Use NumPy-style docstrings for public APIs. Add or update examples when behavior changes; doctest-style examples should remain executable.
- Follow PEP 8 and the repository's 119-character line limit. Format only touched code; do not create repository-wide formatting churn.
- Do not edit generated version files or commit generated documentation under `docs/_build/`.

## Modularity and Code Quality

Modularity is a core design principle of `mlcolvar`. Preserve the separation between data handling, reusable mathematical operations, neural-network components, transforms, estimators, and trainable CV models. Put behavior in the narrowest module that owns it and compose existing components instead of duplicating their logic. Avoid coupling modules through private implementation details or introducing dependencies from lower-level `core` components to task-level `cvs` code.

Simple code is preferred over clever or speculative code. Implement only what the issue or request requires:

- Do not add unrequested features, generalized frameworks, configuration options, compatibility layers, or large refactors.
- Do not write many lines when a short, direct implementation clearly expresses the same behavior.
- Add an abstraction only when it removes meaningful duplication or represents an established concept in the library.
- Keep functions and classes focused on one responsibility. Split logic at real conceptual boundaries, not merely to produce more helpers.
- Prefer explicit control flow and data transformations over hidden side effects or overly compact expressions.
- Use concise, descriptive, and unambiguous names. Names should communicate scientific meaning and tensor roles; avoid one-letter names except for conventional local mathematical notation whose meaning is immediately clear.
- Write code that can be understood without reconstructing the author's intent. Favor readability over premature optimization.
- Add concise comments for non-obvious scientific reasoning, shape assumptions, numerical choices, or constraints. Comments should explain why the code works this way, not narrate self-explanatory syntax.
- Remove obsolete comments when behavior changes, and do not leave commented-out code.

Before expanding the scope of a change, confirm that the extra code is necessary to satisfy the request. When it is not necessary, leave it out.

## Testing

Place tests under `mlcolvar/tests/` using the source path in the filename. For example, changes to `mlcolvar/core/transform/foo.py` belong in `mlcolvar/tests/test_core_transform_foo.py`. Prefer small deterministic tensor fixtures, `pytest.mark.parametrize` for related cases, and `pytest.raises` for error paths.

Run the narrowest relevant test first:

```bash
python -m pytest mlcolvar/tests/test_relevant_module.py -q
python -m pytest mlcolvar/tests/test_relevant_module.py::test_relevant_behavior -q
```

Run the packaged test suite before completing broad or cross-cutting changes:

```bash
python -m pytest -v --pyargs mlcolvar.tests
```

For coverage parity with CI:

```bash
python -m pytest -v --pyargs mlcolvar.tests --cov=mlcolvar --cov-report=xml
```

Notebook changes must execute successfully:

```bash
python -m pytest -v --nbmake docs/notebooks/ --ignore=docs/notebooks/tutorials/data/
```

Documentation changes should build without errors:

```bash
make -C docs html
```

When a full suite is impractical, report exactly which focused checks were run and which broader checks remain.

## Change Checklist

1. Identify the owning module and inspect its nearest implementation and test.
2. Make the smallest change that preserves tensor, Lightning, and export contracts.
3. Add a regression test or feature test alongside the existing test family.
4. Run the focused test, then broaden validation according to the change's scope.
5. Update docstrings, API exports, and user documentation when public behavior changes.