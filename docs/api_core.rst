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

   MSELoss
   ELBOGaussiansLoss
   FisherDiscriminantLoss
   AutocorrelationLoss
   ReduceEigenvaluesLoss
   TDALoss
   CommittorLoss
   GeneratorLoss
   SmartDerivatives

.. rubric:: Stats 

.. currentmodule:: mlcolvar.core.stats

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Stats
   PCA
   LDA 
   TICA

.. rubric:: Transform

.. currentmodule:: mlcolvar.core.transform

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Transform
   

.. rubric:: Transform.descriptors

.. currentmodule:: mlcolvar.core.transform.descriptors

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   PairwiseDistances
   TorsionalAngle
   CoordinationNumbers
   EigsAdjMat
   MultipleDescriptors

.. rubric:: Transform.tools

.. currentmodule:: mlcolvar.core.transform.tools

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   Normalization
   ContinuousHistogram
   SwitchingFunctions