Data
----

General: dataset, module and loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mlcolvar.data

This module contains the classes used for handling datasets and for feeding them to the Lightning trainer.

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   DictDataset
   DictLoader
   DictModule

Graph specific tools
^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: mlcolvar.data.graph

This module contains the classes used for handling and creating graphs.

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   AtomicNumberTable
   Configuration
   get_neighborhood
   create_dataset_from_configurations