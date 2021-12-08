from typing import Optional, Callable, Any, Dict

from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator, \
    Evaluator

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.losses.base import SparkLoss
from lightautoml.tasks import Task as LAMATask
from lightautoml.tasks.base import LAMLMetric, _default_losses, _default_metrics, _valid_task_names


class SparkMetric(LAMLMetric):
    def __init__(
        self,
        name: str,
        evaluator: Evaluator,
        target_col: str,
        prediction_col: str,
        greater_is_better: bool = True,
        metric_params: Optional[Dict] = None
    ):
        """

        Args:
            metric: Specifies metric. Format:
                ``func(y_true, y_false, Optional[sample_weight], **kwargs)`` -> `float`.
            name: Name of metric.
            greater_is_better: Whether or not higher metric value is better.
            one_dim: `True` for single class, False for multiclass.
            weighted: Weights of classes.
            **kwargs: Other parameters for metric.

        """
        self._name = name
        self._evaluator = evaluator
        self._target_col = target_col
        self._prediction_col = prediction_col
        self.greater_is_better = greater_is_better
        self._metric_params = metric_params

    def __call__(self, dataset: SparkDataset, dropna: bool = False):
        assert len(dataset.features) == 1, \
            f"Dataset should contain only one feature that would be interpretated as a prediction"

        prediction_column = dataset.features[0]

        sdf = dataset.data.dropna() if dropna else dataset.data
        sdf = (
            sdf.join(dataset.target, SparkDataset.ID_COLUMN)
                .withColumnRenamed(dataset.target_column, self._target_col)
                .withColumnRenamed(prediction_column, self._prediction_col)
        )

        return self._evaluator.evaluate(sdf, params=self._metric_params)


class Task(LAMATask):

    _default_metrics = {"binary": "areaUnderROC", "reg": "mse", "multiclass": "logLoss"}
    _target_col = "target"
    _prediction_col = "prediction"

    def __init__(
        self,
        name: str,
        loss: Optional[str] = None,
        loss_params: Optional[Dict] = None,
        metric: Optional[str] = None,
        metric_params: Optional[Dict] = None,
        greater_is_better: Optional[bool] = None,
    ):
        super().__init__(name, loss, loss_params, metric, metric_params, greater_is_better)
        assert name in _valid_task_names, "Invalid task name: {}, allowed task names: {}".format(
            name, _valid_task_names
        )

        self._name = name

        # add losses
        # if None - infer from task
        self.losses = {}
        if loss is None:
            loss = _default_losses[self.name]

        if loss_params is None:
            loss_params = {}

        # SparkLoss actualy does nothing, but it is there
        # to male TabularAutoML work
        self.losses = {'lgb': SparkLoss(),'linear_l2': SparkLoss()}

        # TODO: do something with loss, but check at first MLAlgo impl.

        # set callback metric for loss
        # if no metric - infer from task
        if metric is None:
            metric = self._default_metrics[self.name]

        self.metric_params = metric_params if metric_params else dict()

        # TODO: real column names?
        if self._name == "binary":
            self._evaluator = BinaryClassificationEvaluator(metricName=metric,
                                                            rawPredictionCol=self._prediction_col,
                                                            labelCol=self._target_col)
        elif self._name == "reg":
            self._evaluator = RegressionEvaluator(metricName=metric,
                                                  predictionCol=self._prediction_col,
                                                  labelCol=self._target_col)
        else:
            self._evaluator = MulticlassClassificationEvaluator(metricName=metric,
                                                                predictionCol=self._prediction_col,
                                                                labelCol=self._target_col)

        self.metric_name = metric

    def get_dataset_metric(self) -> LAMLMetric:
        return SparkMetric(self.name, self._evaluator, self._target_col, self._prediction_col, self.greater_is_better)
