.. currentmodule:: boxhed_shap

API Reference
=============
This page contains the API reference for public objects and functions in boxhed_shap.
There are also :ref:`example notebooks <api_examples>` available that demonstrate how
to use the API of each object/function.


.. _explanation_api:

Explanation
-----------
.. autosummary::
    :toctree: generated/

    boxhed_shap.Explanation


.. _explainers_api:

explainers
----------
.. autosummary::
    :toctree: generated/

    boxhed_shap.Explainer
    boxhed_shap.explainers.Tree
    boxhed_shap.explainers.GPUTree
    boxhed_shap.explainers.Linear
    boxhed_shap.explainers.Permutation
    boxhed_shap.explainers.Partition
    boxhed_shap.explainers.Sampling
    boxhed_shap.explainers.Additive
    boxhed_shap.explainers.other.Coefficent
    boxhed_shap.explainers.other.Random
    boxhed_shap.explainers.other.LimeTabular
    boxhed_shap.explainers.other.Maple
    boxhed_shap.explainers.other.TreeMaple
    boxhed_shap.explainers.other.TreeGain


.. _plots_api:

plots
-----
.. autosummary::
    :toctree: generated/

    boxhed_shap.plots.bar
    boxhed_shap.plots.waterfall
    boxhed_shap.plots.scatter
    boxhed_shap.plots.heatmap
    boxhed_shap.plots.force
    boxhed_shap.plots.text
    boxhed_shap.plots.image
    boxhed_shap.plots.partial_dependence


.. _maskers_api:

maskers
-------
.. autosummary::
    :toctree: generated/

    boxhed_shap.maskers.Masker
    boxhed_shap.maskers.Independent
    boxhed_shap.maskers.Partition
    boxhed_shap.maskers.Text
    boxhed_shap.maskers.Image


.. _models_api:

models
------
.. autosummary::
    :toctree: generated/

    boxhed_shap.models.Model
    boxhed_shap.models.TeacherForcingLogits


.. _utils_api:

utils
-----
.. autosummary::
    :toctree: generated/

    boxhed_shap.utils.hclust
    boxhed_shap.utils.sample
    boxhed_shap.utils.boxhed_shapley_coefficients
    boxhed_shap.utils.MaskedModel


.. _datasets_api:

datasets
--------
.. autosummary::
    :toctree: generated/

    boxhed_shap.datasets.adult
    boxhed_shap.datasets.boston
    boxhed_shap.datasets.communitiesandcrime
    boxhed_shap.datasets.corrgroups60
    boxhed_shap.datasets.diabetes
    boxhed_shap.datasets.imagenet50
    boxhed_shap.datasets.imdb
    boxhed_shap.datasets.independentlinear60
    boxhed_shap.datasets.iris
    boxhed_shap.datasets.nhanesi
