.. role:: hidden
    :class: hidden-section


sparklightautoml.pipelines.features
==============================

Pipelines for features generation.

Base Classes
-----------------

.. currentmodule:: sparklightautoml.pipelines.features.base

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

.. currentmodule:: sparklightautoml.pipelines.features.lgb_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLGBSimpleFeatures
    SparkLGBAdvancedPipeline


Feature Pipelines for Linear Models
-----------------------------------

.. currentmodule:: sparklightautoml.pipelines.features.linear_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLinearFeatures

Utility Functions
-----------------

.. currentmodule:: sparklightautoml.pipelines.features.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    build_graph