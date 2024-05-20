Collective variables
--------------------

In this section we report the neural network-based collective variables implemented in the library. Note that the linear statistical methods are implemented in ``mlcolvar.core.stats`` instead. 

.. rubric:: Base class

All CVs inherits from this base class, which also implement default methods.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   BaseCV

For each of the specific CV described below there are reported the keys of the expected dataset and the loss function used.

.. rubric:: Unsupervised learning 

CVs based on the autoencoder architecture. Can be used to reconstruct the original input or an arbitrary reference, with an optional reweighting of the data. 

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   AutoEncoderCV
   VariationalAutoEncoderCV

.. rubric:: Supervised learning 

CVs optimized with supervised learning tasks, either classification or regression.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepLDA
   DeepTDA
   RegressionCV

.. rubric:: Time-informed learning

CVs which are optimized on pairs of time-lagged configurations, and optional reweighting for the time-correlation functions.
Note that also the autoencoder-related CVs can fall in this category when the target reference is the time-lagged data.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepTICA

.. rubric:: MultiTask learning

General framework which allows to optimize a single model with different loss functions optimized on different datasets.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   MultiTaskCV

Framework for the numerical determination of the committor function based on its variational principle.

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Committor