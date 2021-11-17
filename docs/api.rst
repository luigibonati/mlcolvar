Documentation
=================

**Base classes**

.. rubric:: Models

.. currentmodule:: mlcvs.models

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   LinearCV
   NeuralNetworkCV

.. rubric:: Estimators

.. currentmodule:: mlcvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   lda.LDA
   tica.TICA

**Collective variables**

.. rubric:: Discriminant-based collective variables

.. currentmodule:: mlcvs.lda

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   LDA_CV
   DeepLDA_CV

.. rubric:: Collective variables as slow modes

.. currentmodule:: mlcvs.tica

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   TICA_CV
   DeepTICA_CV

**Utilities**

.. rubric:: Input/output

.. currentmodule:: mlcvs.utils.io

.. autosummary::
   :toctree: autosummary

   load_dataframe

.. rubric:: Data

.. currentmodule:: mlcvs.utils.data

.. autosummary::
   :toctree: autosummary

   FastTensorDataLoader