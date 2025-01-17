Utils
-----


Input/Output
^^^^^^^^^^^^

Helper functions for loading dataframes (incl. PLUMED files) and directly creating datasets from them.

.. currentmodule:: mlcolvar.utils.io

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   load_dataframe
   create_dataset_from_files


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


Trainer
^^^^^^^
Functions used in conjunction with the lightning Trainer (e.g. logging, metrics...).

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