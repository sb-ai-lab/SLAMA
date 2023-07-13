import logging.config

from examples.spark.examples_utils import get_spark_session, get_dataset
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


feature_pipelines = {
    "linear": SparkLinearFeatures(),
    "lgb_simple": SparkLGBSimpleFeatures(),
    "lgb_adv": SparkLGBAdvancedPipeline()
}


if __name__ == "__main__":
    spark = get_spark_session()

    # settings and data
    cv = 5
    dataset_name = "lama_test_dataset"
    parallelism = 2

    dataset = get_dataset(dataset_name)
    df = spark.read.csv(dataset.path, header=True)

    computations_manager = ParallelComputationsManager(parallelism=parallelism)
    task = SparkTask(name=dataset.task_type)
    reader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)

    ds = reader.fit_read(train_data=df, roles=dataset.roles)

    def build_task(name: str, feature_pipe: SparkFeaturesPipeline):
        def func():
            logger.info(f"Calculating feature pipeline: {name}")
            feature_pipe.fit_transform(ds).data.write.mode('overwrite').format('noop').save()
            logger.info(f"Finished calculating pipeline: {name}")
        return func

    tasks = [build_task(name, feature_pipe) for name, feature_pipe in feature_pipelines.items()]

    computations_manager.session(tasks)
