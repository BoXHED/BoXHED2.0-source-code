.. boxhed_shap documentation master file, created by
   sphinx-quickstart on Tue May 22 10:44:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the boxhed_shap documentation
---------------------------------

.. image:: artwork/boxhed_shap_header.png
   :width: 600px
   :align: center

**boxhed_shap (boxhed_shapley Additive exPlanations)** is a game theoretic approach to explain the output of
any machine learning model. It connects optimal credit allocation with local explanations
using the classic boxhed_shapley values from game theory and their related extensions (see 
`papers <https://github.com/slundberg/boxhed_shap#citations>`_ for details and citations).

Install
=======

boxhed_shap can be installed from either `PyPI <https://pypi.org/project/boxhed_shap>`_ or 
`conda-forge <https://anaconda.org/conda-forge/boxhed_shap>`_::

   pip install boxhed_shap
   or
   conda install -c conda-forge boxhed_shap


Contents
========

.. toctree::
   :maxdepth: 2

   Topical overviews <overviews>
   Tabular examples <tabular_examples>
   Text examples <text_examples>
   Image examples <image_examples>
   Genomic examples <genomic_examples>
   Benchmarks <benchmarks>
   API reference <api>
   API examples <api_examples>
