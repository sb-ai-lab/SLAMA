.. role:: hidden
    :class: hidden-section

slama.ml_algo
===================

Models used for machine learning pipelines.

Base Classes
------------------------

.. currentmodule:: slama.ml_algo.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkTabularMLAlgo
    AveragingTransformer


Available Models
-------------------------

.. currentmodule:: slama.ml_algo

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    linear_pyspark.SparkLinearLBFGS
    boost_lgbm.SparkBoostLGBM

Utilities
-------------------------

.. currentmodule:: slama.ml_algo

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    boost_lgbm.LightGBMModelWrapper
    boost_lgbm.ONNXModelWrapper
