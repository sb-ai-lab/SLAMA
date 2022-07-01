.. role:: hidden
    :class: hidden-section

slama.automl
======================

The main module, which includes the SparkAutoML class, blenders and ready-made presets.

.. currentmodule:: slama.automl.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkAutoML


Presets
-------

Presets for end-to-end model training for special tasks.

.. currentmodule:: slama.automl.presets

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    base.SparkAutoMLPreset
    tabular_presets.SparkTabularAutoML


Blenders
--------

.. currentmodule:: slama.automl.blend

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkBlender
    SparkBestModelSelector
    SparkMeanBlender
    SparkWeightedBlender
