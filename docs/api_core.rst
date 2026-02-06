Core modules
============

These are the building blocks which are used to construct the CVs.

NN
--
This module implements the architectures with learnable weights that can be used to build CV models.

Descriptors-based
^^^^^^^^^^^^^^^^
.. currentmodule:: mlcolvar.core.nn

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   FeedForward

Graphs-based
^^^^^^^^^^^^
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
   CommittorLoss
   GeneratorLoss
   SmartDerivatives


Stats 
-----
This module implements statistical methods with learnable weights that can be used in CV models.

Base class
^^^^^^^^^^
.. currentmodule:: mlcolvar.core.stats

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Stats

Linear methods
^^^^^^^^^^^^^^
.. currentmodule:: mlcolvar.core.stats

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   PCA
   LDA 
   TICA


Transform
---------
This module implements **non-learnable** pre/postprocessing tools 

Base class
^^^^^^^^^^

.. currentmodule:: mlcolvar.core.transform

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Transform
   

Descriptors
^^^^^^^^^^^
This submodule implements several descriptors that can be computed starting from atomic positions.

.. currentmodule:: mlcolvar.core.transform.descriptors

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   PairwiseDistances
   TorsionalAngle
   CoordinationNumbers
   EigsAdjMat
   MultipleDescriptors

Tools
^^^^^
This submodule implements pre/postporcessing tools.

.. currentmodule:: mlcolvar.core.transform.tools

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Normalization
   ContinuousHistogram
   SwitchingFunctions