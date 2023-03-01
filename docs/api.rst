Documentation
=================

Collective variables
--------------------

.. rubric:: Unsupervised learning 

.. currentmodule:: mlcvs.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   AutoEncoder_CV

.. rubric:: Supervised learning 

.. currentmodule:: mlcvs.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Regression_CV
   DeepLDA_CV
   DeepTDA_CV

.. rubric:: Time lagged CVs

.. currentmodule:: mlcvs.cvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DeepTICA_CV


Base classes
------------

These are the building blocks which are used to construct the CVs.

.. rubric:: Stats 

.. currentmodule:: mlcvs.core.stats

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   LDA 
   TICA

.. rubric:: NN

.. currentmodule:: mlcvs.core.nn

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   FeedForward

.. rubric:: Transform

.. currentmodule:: mlcvs.core.transform

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Transform
   Normalization

.. rubric:: Loss

.. currentmodule:: mlcvs.core.loss

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   MSE_loss
   TDA_loss
   reduce_eigenvalues


