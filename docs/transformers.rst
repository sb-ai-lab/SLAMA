.. role:: hidden
    :class: hidden-section


sparklightautoml.transformers
==============================

Basic feature generation steps and helper utils.

Base Classes
------------------------------

.. currentmodule:: sparklightautoml.transformers.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkBaseEstimator
    SparkBaseTransformer
    SparkChangeRolesTransformer
    SparkSequentialTransformer
    SparkUnionTransformer
    SparkColumnsAndRoles
    HasInputRoles
    HasOutputRoles
    ColumnsSelectorTransformer
    DropColumnsTransformer
    PredictionColsTransformer
    ProbabilityColsTransformer



Numeric
------------------------------

.. currentmodule:: sparklightautoml.transformers.numeric

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkFillnaMedianEstimator
    SparkNaNFlagsEstimator
    SparkQuantileBinningEstimator
    SparkStandardScalerEstimator
    SparkFillInfTransformer
    SparkFillnaMedianTransformer
    SparkLogOddsTransformer
    SparkNaNFlagsTransformer
    SparkQuantileBinningTransformer
    SparkStandardScalerTransformer



Categorical
------------------------------

.. currentmodule:: sparklightautoml.transformers.categorical

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLabelEncoderEstimator
    SparkOrdinalEncoderEstimator
    SparkFreqEncoderEstimator
    SparkCatIntersectionsEstimator
    SparkTargetEncoderEstimator
    SparkMulticlassTargetEncoderEstimator
    SparkOHEEncoderEstimator
    SparkLabelEncoderTransformer
    SparkOrdinalEncoderTransformer
    SparkFreqEncoderTransformer
    SparkCatIntersectionsTransformer
    SparkTargetEncoderTransformer
    SparkMultiTargetEncoderTransformer
    SparkCatIntersectionsHelper

Categorical (Scala)
------------------------------

.. currentmodule:: sparklightautoml.transformers.scala_wrappers

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    laml_string_indexer.LAMLStringIndexer
    laml_string_indexer.LAMLStringIndexerModel


Datetime
------------------------------

.. currentmodule:: sparklightautoml.transformers.datetime

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkDateSeasonsEstimator
    SparkTimeToNumTransformer
    SparkBaseDiffTransformer
    SparkDateSeasonsTransformer
    SparkDatetimeHelper