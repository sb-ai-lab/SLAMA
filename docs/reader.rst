.. role:: hidden
    :class: hidden-section


sparklightautoml.reader
=====================

Utils for reading, training and analysing data.

Readers
-------------

.. currentmodule:: sparklightautoml.reader.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkToSparkReader
    SparkToSparkReaderTransformer
    SparkReaderHelper

Utility functions for advanced roles guessing
-------------

.. currentmodule:: sparklightautoml.reader.guess_roles

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    get_category_roles_stat
    get_gini_func
    get_null_scores
    get_numeric_roles_stat
    get_score_from_pipe