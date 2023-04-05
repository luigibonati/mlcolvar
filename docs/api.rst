Documentation
=================

Collective variables
--------------------

.. rubric:: Unsupervised learning 

.. currentmodule:: mlcvs.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   AutoEncoderCV
   VariationalAutoEncoderCV

.. rubric:: Supervised learning 

.. currentmodule:: mlcvs.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   RegressionCV
   DeepLDA
   DeepTDA

.. rubric:: Time lagged CVs

.. currentmodule:: mlcvs.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepTICA


Core modules
------------

These are the building blocks which are used to construct the CVs.

.. rubric:: NN

.. currentmodule:: mlcvs.core.nn

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   FeedForward

.. rubric:: Loss

.. currentmodule:: mlcvs.core.loss

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   mse_loss
   tda_loss
   reduce_eigenvalues_loss
   elbo_gaussians_loss

.. rubric:: Stats 

.. currentmodule:: mlcvs.core.stats

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Stats
   LDA 
   TICA

.. rubric:: Transform

.. currentmodule:: mlcvs.core.transform

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Transform
   Normalization


