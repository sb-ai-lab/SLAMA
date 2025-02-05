import logging
import os
import signal
import sys
import time
from typing import Optional, Any, Dict, Tuple

import psutil
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor
from pyspark.sql import functions as sf

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import log_exec_time
from sparklightautoml.validation.iterators import SparkFoldsIterator

logger = logging.getLogger(__name__)


GENERAL_RUN_PARAMS = {
    'featuresCol': 'Mod_0_LightGBM_vassembler_features',
    'verbosity': 1,
    'useSingleDatasetMode': True,
    'useBarrierExecutionMode': False,
    'isProvideTrainingMetric': True,
    'chunkSize': 10_000,
    'defaultListenPort': 13614,
    'learningRate': 0.03,
    'numLeaves': 64,
    'featureFraction': 0.7,
    'baggingFraction': 0.7,
    'baggingFreq': 1,
    'maxDepth': -1,
    'minGainToSplit': 0.0,
    'maxBin': 255,
    'minDataInLeaf': 5,
    'numIterations': 50,
    'earlyStoppingRound': 200,
    # 'numTasks': None,
    'numThreads': 4,
    'matrixType': 'auto',
    'maxStreamingOMPThreads': 1,

    # 'dataTransferMode': 'bulk',
    # 'numTasks': 6,

    'dataTransferMode': 'streaming',
    # 'numTasks': 6
}


def get_lightgbm_params(spark: SparkSession, dataset_name: str) -> Tuple[str, Dict[str, Any]]:
    data_path = None
    match dataset_name:
        case "company_bankruptcy_dataset":
            dataset_specific_params = {
                'labelCol': "Bankrupt?",
                'objective': 'binary',
                'metric': 'auc',
                'rawPredictionCol': 'raw_prediction',
                'probabilityCol': 'Mod_0_LightGBM_prediction_0',
                'predictionCol': 'prediction',
                'isUnbalance': True
            }
        case "lama_test_dataset":
            dataset_specific_params = {
                'labelCol': 'TARGET',
                'objective': 'binary',
                'metric': 'auc',
                'rawPredictionCol': 'raw_prediction',
                'probabilityCol': 'Mod_0_LightGBM_prediction_0',
                'predictionCol': 'prediction',
                'isUnbalance': True
            }
        case "used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "small_used_cars_dataset":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/small_used_cars_dataset.slama/data.parquet"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "used_cars_dataset_10x":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/used_cars_dataset_10x.slama/data.parquet"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "used_cars_dataset_100x":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/used_cars_dataset_100x.slama/data.parquet"
            # data_path = "hdfs://node21.bdcl:9000/opt/spark_data/used_cars_100x_dataset.csv"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "adv_small_used_cars_dataset":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/adv_small_used_cars_dataset.slama/data.parquet"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "adv_used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case _:
            raise ValueError("Unknown dataset")

    data_path = data_path or f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/CSV/{dataset_name}.csv"

    execs = int(spark.conf.get("spark.executor.instances", "1"))

    return data_path, {
        **GENERAL_RUN_PARAMS,
        **dataset_specific_params,
        "numTasks": execs
    }


def get_spark_session(partitions_num: Optional[int] = None):
    partitions_num = partitions_num if partitions_num else 6

    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
    else:

        extra_jvm_options = "-Dio.netty.tryReflectionSetAccessible=true "

        spark_sess = (
            SparkSession.builder.master(f"local[{partitions_num}]")
            # .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.8")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.8")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.1.jar")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.driver.extraJavaOptions", extra_jvm_options)
            .config("spark.executor.extraJavaOptions", extra_jvm_options)
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            .config("spark.cleaner.referenceTracking", "true")
            .config("spark.cleaner.periodicGC.interval", "1min")
            .config("spark.sql.shuffle.partitions", f"{partitions_num}")
            .config("spark.default.parallelism", f"{partitions_num}")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

    spark_sess.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    # spark_sess.sparkContext.setLogLevel("WARN")

    return spark_sess


def load_data(spark: SparkSession, data_path: str, partitions_coefficient: int = 1) -> DataFrame:
    if data_path.endswith('.csv'):
        data = spark.read.csv(data_path, header=True, inferSchema=True, encoding="UTF-8")
    else:
        data = spark.read.parquet(data_path)

    execs = int(spark.conf.get("spark.executor.instances", "1"))
    cores = int(spark.conf.get("spark.executor.cores", "8"))

    data = data.repartition(execs * cores * partitions_coefficient).cache()
    data.write.mode("overwrite").format("noop").save()

    return data


def load_test_and_train(
    spark: SparkSession, data_path: str, seed: int = 42, test_size: float = 0.2, partitions_coefficient: int = 1
) -> Tuple[DataFrame, DataFrame]:
    assert 0 <= test_size <= 1

    if data_path.endswith('.csv'):
        data = spark.read.csv(data_path, header=True, inferSchema=True, encoding="UTF-8")
    else:
        data = spark.read.parquet(data_path)

    # small adjustment in values making them non-categorial prevent SIGSEGV from happening
    # data = data.na.fill(0.0453)
    # data = data.select(
    #     *[
    #         (sf.col(c) + (sf.rand() / sf.lit(10.0)) + sf.lit(0.05)).alias(c)
    #         for c in data.columns if c not in ['_id', 'price']
    #     ],
    #     'price'
    # )

    execs = int(spark.conf.get("spark.executor.instances", "1"))
    cores = int(spark.conf.get("spark.executor.cores", "8"))

    data = data.repartition(execs * cores * partitions_coefficient).cache()
    data.write.mode("overwrite").format("noop").save()

    train_data, test_data = data.randomSplit([1 - test_size, test_size], seed)

    return train_data, test_data


def clean_java_processes():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        time.sleep(10)
        pids = [proc.pid for proc in psutil.process_iter() if "java" in proc.name()]
        print(f"Found unstopped java processes: {pids}")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except:
                logger.warning(f"Exception during killing the java process with pid {pid}", exc_info=True)


def check_lightgbm(*args):
    dataset_name = args[0]

    print(f"Working with dataset: {dataset_name}")

    spark = get_spark_session()

    data_path, run_params = get_lightgbm_params(spark, dataset_name)

    train_df, test_df = load_test_and_train(spark=spark, data_path=data_path)

    print(f"ASSEMBLED DATASET SIZE: {train_df.count()}")

    features = [c for c in train_df.columns if c not in [run_params['labelCol'], '_id', 'reader_fold_num', 'is_val']]
    assembler = VectorAssembler(inputCols=features, outputCol=run_params['featuresCol'], handleInvalid="keep")

    match run_params['objective']:
        case 'regression':
            lgbm = LightGBMRegressor(**run_params)
        case 'binary':
            lgbm = LightGBMClassifier(**run_params)
        case _:
            raise ValueError()

    df = assembler.transform(train_df)
    model = lgbm.fit(df)
    print("Training is finished")

    df = assembler.transform(test_df)
    predicts_df = model.transform(df)
    predicts_df.write.mode("overwrite").format("noop").save()
    print("Predicting is finished")

    # time.sleep(600)

    spark.stop()
    clean_java_processes()


def check_simple_features_only(*args):
    spark = get_spark_session()

    cv = 5
    # dataset_name = "lama_test_dataset"
    # dataset_name = "small_used_cars_dataset"
    # dataset_name = "used_cars_dataset"
    # dataset_name = "used_cars_dataset_4x"
    # dataset_name = "used_cars_dataset_40x"
    # dataset_name = "company_bankruptcy_dataset"
    # dataset_name = "company_bankruptcy_dataset_100x"
    dataset_name = "company_bankruptcy_dataset_10000x"
    dataset = get_dataset(dataset_name)

    # TODO: there is some problem with composite persistence manager on kubernetes. Need to research later.
    # persistence_manager = get_persistence_manager()
    persistence_manager = PlainCachePersistenceManager()

    with log_exec_time():
        train_df = dataset.load()

        task = SparkTask(dataset.task_type)

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(train_df, roles=dataset.roles, persistence_manager=persistence_manager)

        sdataset = SparkLGBSimpleFeatures().fit_transform(sdataset)
        sdataset.save(f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama", save_mode="overwrite")

        # # How to load
        # sdataset = SparkDataset.load(
        #     path=f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama",
        #     persistence_manager=persistence_manager
        # )
        #
        # print("Dataset Features: ")
        # pprint(sdataset.features)
        # size = sdataset.data.count()
        # print(f"Dataset size: {size}")

    logger.info("Finished")

    spark.stop()


def check_adv_features_only(*args):
    spark = get_spark_session()

    cv = 5
    ml_alg_kwargs = {
        "auto_unique_co": 10,
        "max_intersection_depth": 1,
        "multiclass_te_co": 3,
        "output_categories": False,
        "top_intersections": 4,
    }
    # dataset_name = "lama_test_dataset"
    # dataset_name = "small_used_cars_dataset"
    # dataset_name = "used_cars_dataset"
    # dataset_name = "used_cars_dataset_4x"
    dataset_name = "used_cars_dataset_10x"
    # dataset_name = "used_cars_dataset_40x"
    # dataset_name = "company_bankruptcy_dataset"
    # dataset_name = "company_bankruptcy_dataset_100x"
    # dataset_name = "company_bankruptcy_dataset_10000x"
    dataset = get_dataset(dataset_name)

    persistence_manager = PlainCachePersistenceManager()

    with log_exec_time():
        train_df = dataset.load()

        # optional part
        execs = int(spark.conf.get("spark.executor.instances", "1"))
        cores = int(spark.conf.get("spark.executor.cores", "8"))
        train_df = train_df.repartition(execs * cores * 1)

        task = SparkTask(dataset.task_type)

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False, )
        spipe = SparkLGBAdvancedPipeline(**ml_alg_kwargs)

        sdataset = sreader.fit_read(train_df, roles=dataset.roles, persistence_manager=persistence_manager)
        sdataset = spipe.fit_transform(sdataset)
        name_prefix = "half_adv" if ml_alg_kwargs.get("debug_only_le_without_te", False) else "adv"
        sdataset.save(
            f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{name_prefix}_{dataset_name}.slama",
            save_mode="overwrite",
            num_partitions=1
        )

        # # How to load
        # sdataset = SparkDataset.load(
        #     path=f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama",
        #     persistence_manager=persistence_manager
        # )
        #
        # print("Dataset Features: ")
        # pprint(sdataset.features)
        # size = sdataset.data.count()
        # print(f"Dataset size: {size}")

    logger.info("Finished")

    spark.stop()


def check_lgb_on_prep_dataset(*args):
    spark = get_spark_session()

    dataset_name = os.getenv('DATASET_NAME', 'company_bankruptcy_dataset_100x')
    # dataset_name = "used_cars_dataset"
    dataset = get_dataset(dataset_name)

    # TODO: there is some problem with composite persistence manager on kubernetes. Need to research later.
    # persistence_manager = get_persistence_manager()
    persistence_manager = PlainCachePersistenceManager()

    with log_exec_time():
        sdataset = SparkDataset.load(
            path=f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama",
            persistence_manager=persistence_manager
        )
        score = SparkTask(dataset.task_type).get_dataset_metric()

        iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()

        spark_ml_algo = SparkBoostLGBM(
            default_params={
              "numIterations": 50,
            },
            freeze_defaults=True,
            chunk_size=10_000,
            execution_mode="bulk"
        )

        ml_pipe = SparkMLPipeline(ml_algos=[spark_ml_algo])

        oof_preds_ds = ml_pipe.fit_predict(iterator)
        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

    logger.info("Finished")

    spark.stop()


def check_dataset(*args):
    dataset_name = "adv_small_used_cars_dataset"

    print(f"Working with dataset: {dataset_name}")

    spark = get_spark_session()

    execs = int(spark.conf.get("spark.executor.instances", "1"))
    cores = int(spark.conf.get("spark.executor.cores", "8"))
    partitions_coefficient = 1

    data_path, run_params = get_lightgbm_params(spark, dataset_name)

    df = spark.read.parquet(data_path).repartition(execs * cores * partitions_coefficient).cache()

    print(f"NUM ROWS: {df.count()}")
    # 10_000 * 30_000 == 300_000_000
    scale_coeff = 3_000
    target = 'price'
    num_cols = [c for c in df.columns if c not in ['_id', 'reader_fold_num', target]]

    df = (
        df
        .withColumn('__tmp__', sf.explode(sf.lit(list(range(scale_coeff)))))
        .drop('__tmp__')
        .select(target, *((sf.col(c) + (sf.rand(42) / sf.lit(10.0))).alias(c) for c in num_cols))
    )

    df.write.parquet(
        "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/adv_used_cars_10x.parquet",
        mode="overwrite",
        compression="none"
    )


def main():
    check_name = sys.argv[1]

    match check_name:
        case "lightgbm":
            check_lightgbm(*sys.argv[2:])
        case "check-dataset":
            check_dataset(*sys.argv[2:])
        case "simple-features-only":
            check_simple_features_only(*sys.argv[2:])
        case "adv-features-only":
            check_adv_features_only(*sys.argv[2:])
        case "lgb-on-prep-dataset":
            check_lgb_on_prep_dataset(*sys.argv[2:])
        case _:
            raise ValueError(f"No check with name {check_name}")


if __name__ == "__main__":
    main()
