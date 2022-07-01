.. role:: hidden
    :class: hidden-section


slama.pipelines.features
==============================

Pipelines for features generation.

Base Classes
-----------------

.. currentmodule:: slama.pipelines.features.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkFeaturesPipeline
    SparkTabularDataFeatures
    SparkEmptyFeaturePipeline
    SparkNoOpTransformer
    SelectTransformer
    FittedPipe

Feature Pipelines for Boosting Models
-----------------------------------------

.. currentmodule:: slama.pipelines.features.lgb_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLGBSimpleFeatures
    SparkLGBAdvancedPipeline


Feature Pipelines for Linear Models
-----------------------------------

.. currentmodule:: slama.pipelines.features.linear_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLinearFeatures

Utility Functions
-----------------

.. currentmodule:: slama.pipelines.features.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    build_graph
