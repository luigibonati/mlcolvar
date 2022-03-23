Documentation
=================

The code is structured in a highly modular way, as to make it easier to understand its workings.
The collective variables (CVs) presented here are constructed combining a **model** and an **estimator**.

* The model can be chosen to be a linear combination of descriptors or a non linear transformation operated by a neural-network.
* The estimators implemented are Fisher's discriminant (LDA) and Time-lagged independent component (TICA). While the former allows to devise CVs as the variables which most discriminate between a given set of states, the latter is used to extract CVs as the slowly decorrelating modes of a sampling dynamics. 

These combination give rise to the different CVs which have been proposed in the literature: `(H)LDA_CV, DeepLDA_CV, TICA_CV,` and `DeepLDA_CV`.

Finally, a few utilities are implemented to handle input/output of data and to efficiently train the CVs.

Base classes
------------

.. rubric:: Models

These are the base models which are used to construct the CVs. 
To keep consistency between them, both are derived from ``torch.nn.Module``. 
The weights of the NN are saved as parameters, while the coefficients of the linear combinations are saved as buffers.

.. currentmodule:: mlcvs.models (``mlcvs.models``)

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   mlcvs.models.LinearCV
   mlcvs.models.NeuralNetworkCV

.. rubric:: Estimators

The following classes implement the calculation of the estimators used to optimize the parameters.

.. currentmodule:: mlcvs

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   lda.LDA
   tica.TICA

Collective variables
--------------------

The CVs are implemented by combining a given model (from which they inherit from) and an estimator (saved as member). They are divided in two families, based on whether they are built to discriminate between the a set of given states or to approximate the slow modes of a sampling dynamics.

.. rubric:: Discriminant-based collective variables (``mlcvs.lda``)

.. currentmodule:: mlcvs.lda

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   LDA_CV
   DeepLDA_CV

.. rubric:: Collective variables as slow modes (``mlcvs.tica``)

.. currentmodule:: mlcvs.tica

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   TICA_CV
   DeepTICA_CV

Utilities
---------

Some helper functions are defined to handle to input/output of the data, with particular emphasis to compatibility with PLUMED output files. Furthermore, to assist the training of the neural-networks based CVs, some objects and functions are defined to assist in the creation of datasets for efficients training and to help with the optimization process.

.. rubric:: Input/output (``mlcvs.utils.io``)

.. currentmodule:: mlcvs.utils.io

.. autosummary::
   :toctree: autosummary

   load_dataframe
   plumed_to_pandas

.. rubric:: Dataset/dataloaders (``mlcvs.utils.data``)

.. currentmodule:: mlcvs.utils.data

.. autosummary::
   :toctree: autosummary

   FastTensorDataLoader
   create_time_lagged_dataset

.. rubric:: Optimization (``mlcvs.utils.optim``)

Convenience functions to adjust the training of the neural networks based on a validation score. When the latter does not decrease anymore the simulation can be interrputed (``EarlyStopping``) or the learning rate decreased (``LRScheduler``). Note that the learning rate scheduler is not implemented yet in the `fit` functions of the CVs.

.. currentmodule:: mlcvs.utils.optim

.. autosummary::
   :toctree: autosummary

   EarlyStopping
   LRScheduler

.. rubric:: FES (``mlcvs.utils.fes``)

.. currentmodule:: mlcvs.utils.fes

.. autosummary::
   :toctree: autosummary

   compute_fes

.. rubric:: Plot (``mlcvs.utils.plot``)

Helper functions to plot the results with Matplotlib. A custom palette is also defined (`FESSA <https://github.com/luigibonati/fessa-color-palette>`_). Upon loading ``mlcvs`` it is possible to use this color scheme direclly with Matplotlib. This has been designed for free energy landscapes visualization.

   