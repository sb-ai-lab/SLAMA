from examples.spark.examples_utils import get_spark_session, get_dataset
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask


feature_pipelines = {
    "linear": SparkLinearFeatures(),
    "lgb_simple": SparkLGBSimpleFeatures(),
    "lgb_adv": SparkLGBAdvancedPipeline()
}


if __name__ == "__main__":
    spark = get_spark_session()

    # settings and data
    cv = 5
    feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
    dataset_name = "lama_test_dataset"
    dataset = get_dataset(dataset_name)
    df = spark.read.csv(dataset.path, header=True)

    task = SparkTask(name=dataset.task_type)
    reader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
    feature_pipe = feature_pipelines.get(feat_pipe, None)

    assert feature_pipe is not None, f"Unsupported feat pipe {feat_pipe}"

    ds = reader.fit_read(train_data=df, roles=dataset.roles)
    ds = feature_pipe.fit_transform(ds)

    # save processed data
    ds.save(f"/tmp/{dataset_name}__{feat_pipe}__features.dataset", save_mode='overwrite')
