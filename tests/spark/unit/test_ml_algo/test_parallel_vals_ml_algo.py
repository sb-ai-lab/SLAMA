from lightautoml.utils.timer import PipelineTimer
from pyspark.sql import SparkSession

from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import make_spark, spark as spark_sess, dataset as spark_dataset

make_spark = make_spark
spark = spark_sess
dataset = spark_dataset

ml_alg_kwargs = {
    'auto_unique_co': 10,
    'max_intersection_depth': 3,
    'multiclass_te_co': 3,
    'output_categories': True,
    'top_intersections': 4
}


def test_parallel_timer_exceeded(spark: SparkSession, dataset: SparkDataset):
    """
    Only two folds of five can be computed under conditions provided to the timer.
    The code should not raise any exception.
    """
    pipeline_timer = PipelineTimer(timeout=2)
    pipeline_timer.start()

    tv_iter = SparkFoldsIterator(dataset)
    ml_algo = SparkBoostLGBM(
        timer=pipeline_timer.get_task_timer(),
        computations_settings=ParallelComputationsManager(parallelism=2)
    )
    oof_preds = ml_algo.fit_predict(tv_iter)

    oof_preds.data.write.mode("overwrite").format("noop").save()
