Utils
-----

Input/Output
^^^^^^^^^^^^

Helper functions for loading dataframes (including PLUMED files) and directly creating datasets from files or trajectories.

.. currentmodule:: mlcolvar.io

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   load_dataframe


Time-lagged datasets
^^^^^^^^^^^^^^^^^^^^

Create a dataset of pairs of time-lagged configurations.

.. currentmodule:: mlcolvar.utils.timelagged

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   create_timelagged_dataset


FES
^^^

Compute (and plot) the free energy surface along the CVs.

.. currentmodule:: mlcolvar.utils.fes

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   compute_fes


AOT compilation
^^^^^^^^^^^^^^^

Utilities for exporting and loading graph-based CV models using PyTorch Ahead-of-Time (AOT) compilation.

.. currentmodule:: mlcolvar.utils.aot

.. autosummary::
   :toctree: autosummary

   export
   load

Graph utilities
"""""""""""""""

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   GraphAdapter


Trainer
^^^^^^^

Functions used in conjunction with the Lightning Trainer (e.g. logging and metrics).

.. currentmodule:: mlcolvar.utils.trainer

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   MetricsCallback


Plot
^^^^

.. currentmodule:: mlcolvar.utils.plot

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   plot_metrics
   plot_features_distribution
