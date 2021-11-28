Documentation
=================

The code is structured in a highly modular way, as to make it easier to understand its workings.

The collective variables (CVs) presented here are constructed combining a `model` and an `estimator`.

* The model can be chosen to be a linear combination of descriptors or a non linear transformation operated by a neural-network.
* The estimators implemented are Fisher's discriminant (LDA) and Time-lagged independent component (TICA). While the former allows to devise CVs as the variables which most discriminate between a given set of states, the latter is used to extract CVs as the slowly decorrelating modes of a sampling dynamics. 

These combination give rise to the different CVs which have been proposed in the literature: (H)LDA_CV, DeepLDA_CV, TICA_CV, and DeepLDA_CV.

Finally, a few utilities are implemented to handle input/output of data and to efficiently train the CVs.

**Base classes**

.. rubric:: Models

These are the base models which are used to construct the CVs. 
To keep consistency between them, both are derived from `torch.nn.Module`. 
The weights of the NN are saved as parameters, while the coefficients of the linear combinations are saved as buffers.

.. currentmodule:: mlcvs.models

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   LinearCV
   NeuralNetworkCV

.. rubric:: Estimators

The following classes implement the calculation of the estimators used to optimize the parameters.

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
   create_time_lagged_dataset