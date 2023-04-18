Documentation
=================

Collective variables
--------------------

.. rubric:: Unsupervised learning 

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   AutoEncoderCV
   VariationalAutoEncoderCV

.. rubric:: Supervised learning 

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   RegressionCV
   DeepLDA
   DeepTDA

.. rubric:: Time lagged CVs

.. currentmodule:: mlcolvar.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepTICA


Core modules
------------

These are the building blocks which are used to construct the CVs.

.. rubric:: NN

.. currentmodule:: mlcolvar.core.nn

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   FeedForward

.. rubric:: Loss

.. currentmodule:: mlcolvar.core.loss

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   mse_loss
   tda_loss
   reduce_eigenvalues_loss
   elbo_gaussians_loss

.. rubric:: Stats 

.. currentmodule:: mlcolvar.core.stats

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Stats
   LDA 
   TICA

.. rubric:: Transform

.. currentmodule:: mlcolvar.core.transform

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Transform
   Normalization


