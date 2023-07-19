import functools
import logging
import os
import pickle
import shutil
from multiprocessing.pool import ThreadPool
from typing import Tuple, List, Dict, Any

# noinspection PyUnresolvedReferences
from pyspark import inheritable_thread_target
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from examples.spark.examples_utils import get_spark_session, get_dataset, prepare_test_and_train
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.transformers.scala_wrappers.balanced_union_partitions_coalescer import \
    BalancedUnionPartitionsCoalescerTransformer
from sparklightautoml.utils import log_exec_timer

logger = logging.getLogger(__name__)


params = {
    'learningRate': 0.01,
    'numLeaves': 32,
    'featureFraction': 0.7,
    'baggingFraction': 0.7,
    'baggingFreq': 1,
    'maxDepth': -1,
    'minGainToSplit': 0.0,
    'maxBin': 255,
    'minDataInLeaf': 5,
    'numIterations': 3000,
    'earlyStoppingRound': 200,
    'objective': 'binary',
    'metric': 'auc'
}


class ParallelExperiment:
    def __init__(self,
                 spark: SparkSession,
                 dataset_name: str,
                 partitions_num: int = 4,
                 seed: int = 42,
                 cv: int = 5):
        self.spark = spark
        self.dataset_name = dataset_name
        self.partitions_num = partitions_num
        self.seed = seed
        self.cv = cv
        self.base_dataset_path = f"/opt/spark_data/parallel_slama_{dataset_name}"
        self.train_path = os.path.join(self.base_dataset_path, "train.parquet")
        self.test_path = os.path.join(self.base_dataset_path, "test.parquet")
        self.metadata_path = os.path.join(self.base_dataset_path, "metadata.pickle")

    def prepare_dataset(self, force=True):
        logger.info(f"Preparing dataset {self.dataset_name}. "
                    f"Writing train, test and metadata to {self.base_dataset_path}")

        if os.path.exists(self.base_dataset_path) and not force:
            logger.info(f"Found existing {self.base_dataset_path}. Skipping writing dataset files")
            return
        elif os.path.exists(self.base_dataset_path):
            logger.info(f"Found existing {self.base_dataset_path}. "
                        f"Removing existing files because force is set to True")
            shutil.rmtree(self.base_dataset_path)

        dataset = get_dataset(self.dataset_name)

        train_df, test_df = prepare_test_and_train(dataset, self.seed)

        task = SparkTask(dataset.task_type)

        sreader = SparkToSparkReader(task=task, cv=self.cv, advanced_roles=False)
        spark_features_pipeline = SparkLGBSimpleFeatures()

        # prepare train
        train_sdataset = sreader.fit_read(train_df, roles=dataset.roles)
        train_sdataset = spark_features_pipeline.fit_transform(train_sdataset)

        # prepare test
        test_sdataset = sreader.read(test_df, add_array_attrs=True)
        test_sdataset = spark_features_pipeline.transform(test_sdataset)

        os.makedirs(self.base_dataset_path)

        train_sdataset.data.write.parquet(self.train_path)
        test_sdataset.data.write.parquet(self.test_path)

        metadata = {
            "roles": train_sdataset.roles,
            "task_type": dataset.task_type,
            "target": dataset.roles["target"]
        }

        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Dataset {self.dataset_name} has been prepared.")

    @property
    def train_dataset(self) -> DataFrame:
        return self.spark.read.parquet(self.train_path)

    @property
    def test_dataset(self) -> DataFrame:
        return self.spark.read.parquet(self.test_path)

    @property
    def metadata(self) -> Dict[str, Any]:
        with open(self.metadata_path, "rb") as f:
            return pickle.load(f)

    def train_model(self, fold: int) -> Tuple[int, float]:
        logger.info(f"Starting to train the model for fold #{fold}")

        train_df = self.train_dataset
        test_df = self.test_dataset
        md = self.metadata
        task_type = md["task_type"]

        train_df.sql_ctx.sparkSession.sparkContext.setLocalProperty("spark.scheduler.mode", "FAIR")

        prediction_col = 'LightGBM_prediction_0'
        if task_type in ["binary", "multiclass"]:
            params["rawPredictionCol"] = 'raw_prediction'
            params["probabilityCol"] = prediction_col
            params["predictionCol"] = 'prediction'
            params["isUnbalance"] = True
        else:
            params["predictionCol"] = prediction_col

        if task_type == "reg":
            params["objective"] = "regression"
            params["metric"] = "mse"
        elif task_type == "binary":
            params["objective"] = "binary"
            params["metric"] = "auc"
        elif task_type == "multiclass":
            params["objective"] = "multiclass"
            params["metric"] = "multiclass"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if task_type != "reg":
            if "alpha" in params:
                del params["alpha"]
            if "lambdaL1" in params:
                del params["lambdaL1"]
            if "lambdaL2" in params:
                del params["lambdaL2"]

        assembler = VectorAssembler(
            inputCols=list(md['roles'].keys()),
            outputCol="LightGBM_vassembler_features",
            handleInvalid="keep"
        )

        lgbm_booster = LightGBMRegressor if task_type == "reg" else LightGBMClassifier

        lgbm = lgbm_booster(
            **params,
            featuresCol=assembler.getOutputCol(),
            labelCol=md['target'],
            validationIndicatorCol='is_val',
            verbosity=1,
            useSingleDatasetMode=True,
            isProvideTrainingMetric=True,
            chunkSize=4_000_000,
            useBarrierExecutionMode=True,
            numTasks=2,
            numThreads=2
        )

        if task_type == "reg":
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        train_df = train_df.withColumn('is_val', sf.col('reader_fold_num') == fold)

        valid_df = train_df.where('is_val')
        train_df = train_df.where(~sf.col('is_val'))
        full_data = valid_df.unionByName(train_df)
        full_data = BalancedUnionPartitionsCoalescerTransformer().transform(full_data)

        transformer = lgbm.fit(assembler.transform(full_data))
        preds_df = transformer.transform(assembler.transform(test_df))

        logger.info(f"Props #{fold}:"
                    f" {full_data.sql_ctx.sparkSession.sparkContext.getLocalProperty('spark.task.cpus')}")

        score = SparkTask(task_type).get_dataset_metric()
        metric_value = score(
            preds_df.select(
                SparkDataset.ID_COLUMN,
                sf.col(md['target']).alias('target'),
                sf.col(prediction_col).alias('prediction')
            )
        )

        logger.info(f"Finished training the model for fold #{fold}")

        return fold, metric_value

    def run(self, parallelism: int = 3) -> List[Tuple[int, float]]:
        with log_exec_timer("Parallel experiment runtime"):
            logger.info("Starting to run the experiment")

            tasks = [
                functools.partial(
                    self.train_model,
                    fold
                )
                for fold in range(parallelism)
            ]

            pool = ThreadPool(processes=parallelism)
            tasks = map(inheritable_thread_target, tasks)
            results = (result for result in pool.imap_unordered(lambda f: f(), tasks) if result)
            results = sorted(results, key=lambda x: x[0])

            logger.info("The experiment is finished")
            return results


def main():
    partitions_num = 6
    spark = get_spark_session(partitions_num=partitions_num)

    exp = ParallelExperiment(spark, dataset_name="used_cars_dataset")
    exp.prepare_dataset()
    results = exp.run()

    for fold, metric_value in results:
        logger.info(f"Metric value (fold = {fold}): {metric_value}")

    spark.stop()


if __name__ == "__main__":
    main()
