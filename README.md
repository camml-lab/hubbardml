# HubbardML


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/muhrin/hubbardml/?ref=repository-badge)

## About

This repository contains source code for our machine learning model for predicting self-consistent Hubbard parameters, as presented in this work:

Uhrin, M., Zadoks, A., Binci, L., Marzari, N., & Timrov, I. (2024). Machine learning Hubbard parameters with equivariant neural networks. http://arxiv.org/abs/2406.02457

The experiments carried out in this work can be found in the `experiments/` folder along with all the notebooks to generate the plots.

As an example, from experiments you can use:

`python run.py experiment=predict_hp model=u`

to run an experiment that trains a model to predict Hubbard U values from a linear-response dataset.

Additional experiments can be found in the `experiments/experiment/` folder.
