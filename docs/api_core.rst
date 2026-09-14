Core modules
============

These are the building blocks used to construct CVs.


NN
--

This module implements neural-network architectures with learnable weights
that can be used to build CV models.


Descriptor-based
^^^^^^^^^^^^^^^^

.. currentmodule:: mlcolvar.core.nn

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   FeedForward


Graph-based
^^^^^^^^^^^

.. currentmodule:: mlcolvar.core.nn.graph

Base class
""""""""""

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   BaseGNN


Architectures
"""""""""""""

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   SchNetModel
   PaiNNModel
   GVPModel


Loss
----

This module implements the loss functions that can be used to optimize CV models.

.. currentmodule:: mlcolvar.core.loss

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   MSELoss
   ELBOGaussiansLoss
   FisherDiscriminantLoss
   AutocorrelationLoss
   ReduceEigenvaluesLoss
   TDALoss
   ContrastiveLoss
   CommittorLoss
   GeneratorLoss
   SmartDerivatives


Estimators
----------

This module implements statistical estimators used in CV models.


Base class
^^^^^^^^^^

.. currentmodule:: mlcolvar.core.estimators

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Estimator


Linear methods
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   PCA
   LDA
   TICA
   Generator


Transform
---------

This module implements **non-learnable** pre- and post-processing tools.


Base class
^^^^^^^^^^

.. currentmodule:: mlcolvar.core.transform

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Transform


Descriptors
^^^^^^^^^^^

This submodule implements descriptors that can be computed from atomic
positions.

.. currentmodule:: mlcolvar.core.transform.descriptors

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   PairwiseDistances
   TorsionalAngles
   CoordinationNumbers
   EigsAdjMat
   MultipleDescriptors


Tools
^^^^^

This submodule implements pre- and post-processing tools.

.. currentmodule:: mlcolvar.core.transform.tools

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Normalization
   ContinuousHistogram
   SwitchingFunctions