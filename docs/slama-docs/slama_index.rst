SLAMA (Spark version of LAMA)
==================

This is a distributed version of LAMA library written on Spark framework.
SLAMA brings LAMA functionality on Spark including:
- Automatic hyperparameter tuning, data processing.
- Automatic typing, feature selection.
- Automatic time utilization.
- Automatic report creation.
- Easy-to-use modular scheme to create your own pipelines.
- Support of Spark ML pipelines, including saving/loading.
- Caching and checkpointing of intermediate results

Known limitations:
- Only the tabular preset is currently supported

.. toctree::
    :maxdepth: 1
    :caption: Python API

    automl
    dataset
    ml_algo
    pipelines
    pipelines.selection
    pipelines.features
    pipelines.ml
    reader
    report
    tasks
    transformers
    validation
