Machine Learning Collective Variables for Enhanced Sampling
=================================================

[//]: # (Badges)
**CODE**  [![Documentation Status](https://readthedocs.org/projects/mlcolvar/badge/?version=latest)](https://mlcolvar.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Build Status](https://github.com/luigibonati/mlcolvar/workflows/CI/badge.svg)](https://github.com/luigibonati/mlcolvar/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/luigibonati/mlcolvar/branch/main/graph/badge.svg?token=H01H68KNNG)](https://codecov.io/gh/luigibonati/mlcolvar)
[![license](https://img.shields.io/github/license/luigibonati/mlcolvar)](https://github.com/luigibonati/mlcolvar/blob/main/LICENSE)

**PAPER**  [![paper](https://img.shields.io/badge/doi-10.1063/5.0156343-blue)](https://doi.org/10.1063/5.0156343)
[![preprint](https://img.shields.io/badge/arXiv-2305.19980-red)](https://arxiv.org/abs/2305.19980)

---


<img src="https://raw.githubusercontent.com/luigibonati/mlcolvar/main/docs/images/logo_name_black_big.png" width="400" />


`mlcolvar` is a Python library aimed to help design data-driven collective-variables (CVs) for enhanced sampling simulations. The key features are:

1. A unified framework to help test and use (some) of the CVs proposed in the literature. 
2. A modular interface to simplify the development of new approaches and the contamination between them.
3. A streamlined distribution of CVs in the context of advanced sampling. 

The library is built upon the [PyTorch](https://pytorch.org/) ML library as well as the [Lightning](https://lightning.ai/) high-level framework. 

---

Some of the **CVs** which are implemented, organized by learning setting:
- _Unsupervised_: PCA, (Variational) AutoEncoders [[1](http://dx.doi.org/%2010.1002/jcc.25520),[2](http://dx.doi.org/%2010.1021/acs.jctc.1c00415)]
- _Supervised_: LDA [[3](http://dx.doi.org/10.1021/acs.jpclett.8b00733)], DeepLDA [[4](http://dx.doi.org/%2010.1021/acs.jpclett.0c00535)], DeepTDA [[5](http://dx.doi.org/%2010.1021/acs.jpclett.1c02317)]
- _Time-informed_: TICA [[6](http://dx.doi.org/%2010.1063/1.4811489)], DeepTICA/SRVs [[7](http://dx.doi.org/10.1073/pnas.2113533118),[8](http://dx.doi.org/%2010.1063/1.5092521)], VDE [[9](http://dx.doi.org/10.1103/PhysRevE.97.062412)]

And many others can be implemented based on the building blocks or with simple modifications. Check out the documentation and the examples section!

---

**PLUMED interface**: the resulting CVs can be deployed for enhancing sampling with the [PLUMED](https://www.plumed.org/) package via the [pytorch](https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html>`_) interface, available since version 2.9. 

---

**Notes**: in early versions (`v<=0.2.*`) the library was called `mlcvs`. This is still accessible for compatibility with PLUMED masterclasses in the [releases](https://github.com/luigibonati/mlcolvar/releases) or by cloning the `pre-lightning` branch.

---

Copyright (c) 2023 Luigi Bonati, Enrico Trizio, Andrea Rizzi and Michele Parrinello. 
Structure of the project is based on 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms).
