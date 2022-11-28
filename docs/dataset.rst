.. role:: hidden
    :class: hidden-section

sparklightautoml.dataset
===================

Provides base entities for working with data.

Dataset Interfaces
-------------------

.. currentmodule:: sparklightautoml.dataset

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    base.SparkDataset
    base.PersistenceLevel
    base.PersistenceManager
    base.Unpersistable

Roles
-----------

Role contains information about the column, which determines how it is processed.

.. currentmodule:: sparklightautoml.dataset.roles

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    NumericVectorOrArrayRole

Persistence
-----------

Persistence managers are responsible for caching and storing intermediate results on various steps during AutoML process.
Storing intermediate results is required by various reasons.

Depending on the manager, it can be used for the following goals:
* Support iterative data processing and preventing repeatition of calculations
* Prunning of long plans that slows down catalyst optimizer
* Prunning of long lineages that increase overheads on tasks serialization (and may lead to large broadcasts)
* Creating reliable checkpoints preventing long recalculations in case of failures
* Optimize joins converting them into shuffle-less merge joins instead of SortMerge joins
(optimization of split-merge patterns in the process of multiple models/multiple feature generation)

For instance, PlainCachePersistenceManager allows to support iterative data processing and provides fast storing
due to leveraging Spark caching mechanism which may employ RAM, but cannot provide for the rest of goals.
From the other side, BucketedPersistenceManager can deliver for all the goals, but requires more time to store data due
to writing to external storage like HDFS. LocalCheckpointPersistenceManager is in the middle:
it can deliver only the first three goals, but store data fast leveraging RAM and DISK if necessary

Different persistence managers may be of more usefulness depending on current step in the automl process.
There can be found several explicit levels of storing stated in PersistenceLevel entity:
* READER level marks the beginning of the automl process, root of all pipelines, executed only onces.
* REGULAR means storing data somewhere in the middle of ML pipeline, mainly feature processing or model training.
* CHECKPOINT is used for denoting data storing in the very end of ML pipeline.
These data will consist only of predictions made by one or several ML models thus making the dataframe being stored relatively small.

All persistence managers can be divided on two main types depending on how they handle different levels
supplied during calling .persist():
* simple managers, that exploit the same approach to store intermediate results on all levels
* composite managers (their name starts with 'Composite' prefix) that can employ different approaches
to store data for different levels.

CompositeBucketedPersistenceManager should be used in most cases. It creates a bucketed dataset on READER level,
which is an expensive operation executed only once. In exchanges, it leads to making all joins
(on the direct descendants of the main dataframe) in the downstream process to be either broadcast joins or merge joins.
In both cases it wouldn't require shuffle. On REGULAR level, mainly for the sake of supporting fast iterative data processing,
it employs PlainCachePersistenceManager. On CHECKPOINT level, having a relatively small dataframe after the end of heavy
data processing and computations this manager opts to reliable data storing using BucketedPersistenceManager.
This choice is also motivated by prunning of long plan and lineage which have grown large up to this moment.

CompositePlainCachePersistenceManager uses PlainCachePersistenceManager for READER and REGULAR levels,
avoiding expensive initial creation of a bucketed dataset. On CHECKPOINT level, it relies on BucketedPersistenceManager
with the same motivation as for the previous case. However, it does have some advantages it should be used with caution.
Use cases when it may be used requires specific Spark Session and AutoML configurations having the following traits:
* AutoML has only one level of ML pipelines or two levels with skip_conn=False
* autoBroadcastJoinThreshold is set high enough to handle some minor joins
The alternative case:
* AutoML has two levels with skip_conn=True
* autoBroadcastJoinThreshold is set sufficiently high to make joining the main dataframe with resulting dataframes
from the first level (containing predictions) shuffle-less

These conditions may change in the future.

.. currentmodule:: sparklightautoml.dataset.persistence

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    BasePersistenceManager
    PlainCachePersistenceManager
    LocalCheckpointPersistenceManager
    BucketedPersistenceManager
    CompositePersistenceManager
    CompositeBucketedPersistenceManager
    CompositePersistenceManager