.. mlcolvar documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mlcolvar: Machine Learning Collective Variables
===============================================

``mlcolvar``, short for Machine Learning COLlective VARiables, is a Python library aimed to help design data-driven collective-variables (CVs) for enhanced sampling simulations.

The **goals** of `mlcolvar` are to develop:

1. A unified framework to help test and use (some) of the CVs proposed in the literature. 
2. A modular interface to simplify the development of new approaches and the contamination between them.
3. A streamlined distribution of CVs in the context of advanced sampling (using PLUMED). 

The library is built upon the `PyTorch <https://pytorch.org/>`_ ML library as well as the `Lightning <https://lightning.ai/>`_ high-level framework. 

The **workflow** for training CVs is illustrated in the figure below. The resulting CVs can be then deployed for enhancing sampling with the `PLUMED <https://www.plumed.org/>`_ package via the `pytorch <https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html>`_ interface. 

.. image:: notebooks/tutorials/images/graphical_overview_mlcvs.png
  :width: 800
  :alt: Example workflow 

Table of contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   tutorials
   plumed

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
