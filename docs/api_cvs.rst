Collective variables
--------------------

This section lists the neural-network-based collective variables implemented
in the library. Linear statistical methods are instead implemented in
``mlcolvar.core.estimators``.


.. rubric:: Base class

All CVs inherit from this base class, which also implements common methods
shared across different CV models.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   BaseCV

For each CV listed below, the corresponding API documentation describes the
expected dataset structure and the loss function used for training.


.. rubric:: Unsupervised learning

CVs based on autoencoder architectures. These models can be used to reconstruct
the original input or an arbitrary target, with optional reweighting of the
training data.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   AutoEncoderCV
   VariationalAutoEncoderCV


.. rubric:: Supervised learning

CVs optimized using supervised-learning tasks, including classification and
regression.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepLDA
   DeepTDA
   RegressionCV


.. rubric:: Time-informed learning

CVs optimized using pairs of time-lagged configurations, with optional
reweighting of the time-correlation functions.

Autoencoder-based CVs can also be used in this setting when the reconstruction
target is a time-lagged configuration.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepTICA
   SelfTICA


.. rubric:: Multi-task learning

General framework for optimizing a single model using multiple loss functions
and potentially different datasets.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   MultiTaskCV


.. rubric:: Committor

Framework for the numerical determination of the committor function based on
its variational principle.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Committor


.. rubric:: Infinitesimal-generator learning

CVs designed to learn eigenfunctions of the infinitesimal generator from
weighted configurations and spatial derivatives, without requiring explicit
time-lagged configuration pairs.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepGenerator