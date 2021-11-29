import logging
from typing import Tuple, cast, List, Optional

import numpy as np
from pyspark.ml import Model
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.sql import functions as F

from lightautoml.ml_algo.base import MLAlgo
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.utils.timer import TaskTimer
from lightautoml.validation.base import TrainValidIterator

logger = logging.getLogger(__name__)


class TabularMLAlgo(MLAlgo):
    """Machine learning algorithms that accepts numpy arrays as input."""

    _name: str = "TabularAlgo"
    _default_prediction_column_name: str = "prediction"

    def __init__(
            self,
            default_params: Optional[dict] = None,
            freeze_defaults: bool = True,
            timer: Optional[TaskTimer] = None,
            optimization_search_space: Optional[dict] = {},
    ):
        super().__init__(default_params, freeze_defaults, timer, optimization_search_space)
        self.n_classes: Optional[int] = None

    @property
    def prediction_column(self) -> str:
        return self._default_prediction_column_name

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> SparkDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``numpy.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """
        self.timer.start()

        assert self.is_fitted is False, "Algo is already fitted"
        # init params on input if no params was set before
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)

        iterator_len = len(train_valid_iterator)
        if iterator_len > 1:
            logger.info("Start fitting \x1b[1m{}\x1b[0m ...".format(self._name))
            logger.debug(f"Training params: {self.params}")

        # save features names
        self._features = train_valid_iterator.features
        # get metric and loss if None
        self.task = train_valid_iterator.train.task

        preds_ds = cast(SparkDataset, train_valid_iterator.get_validation_data())

        # spark
        outp_dim = 1
        if self.task.name == "multiclass":
            # TODO: SPARK-LAMA working with target should be reflected in SparkDataset
            tdf: SparkDataFrame = preds_ds.target
            outp_dim = tdf.select(F.max(preds_ds.target_column).alias("max")).first()
            outp_dim = outp_dim["max"] + 1

        self.n_classes = outp_dim

        # preds_arr = np.zeros((preds_ds.shape[0], outp_dim), dtype=np.float32)
        # counter_arr = np.zeros((preds_ds.shape[0], 1), dtype=np.float32)

        preds_dfs: List[SparkDataFrame] = []

        pred_col_prefix = "prediction"

        # TODO: SPARK-LAMA - we need to cache the "parent" dataset of the train_valid_iterator
        # train_valid_iterator.cache()

        # TODO: Make parallel version later
        for n, (idx, train, valid) in enumerate(train_valid_iterator):
            if iterator_len > 1:
                logger.info2(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m =====".format(n, self._name)
                )
            self.timer.set_control_point()

            model, pred, prediction_column = self.fit_predict_single_fold(train, valid)
            pred = pred.select(
                SparkDataset.ID_COLUMN,
                F.col(prediction_column).alias(f"{pred_col_prefix}_{n}")
            )
            self.models.append(model)
            preds_dfs.append(pred)

            self.timer.write_run_info()

            if (n + 1) != len(train_valid_iterator):
                # split into separate cases because timeout checking affects parent pipeline timer
                if self.timer.time_limit_exceeded():
                    logger.info("Time limit exceeded after calculating fold {0}\n".format(n))
                    break

        # combine predictions of all models and make them into the single one
        full_preds_df = self._average_predictions(preds_ds, preds_dfs, pred_col_prefix)

        # TODO: send the "parent" dataset of the train_valid_iterator for unwinding later
        #       e.g. from the train_valid_iterator
        preds_ds = self._set_prediction(preds_ds, full_preds_df)

        if iterator_len > 1:
            logger.info(
                f"Fitting \x1b[1m{self._name}\x1b[0m finished. score = \x1b[1m{self.score(preds_ds)}\x1b[0m")

        if iterator_len > 1 or "Tuned" not in self._name:
            logger.info("\x1b[1m{}\x1b[0m fitting and predicting completed".format(self._name))
        return preds_ds

    def fit_predict_single_fold(self, train: SparkDataset, valid: SparkDataset) -> Tuple[Model, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        raise NotImplementedError

    def predict_single_fold(self, model: Model, dataset: SparkDataset) -> SparkDataFrame:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: Dataset used for prediction.

        Returns:
            Predictions for input dataset.

        """
        raise NotImplementedError

    def predict(self, dataset: SparkDataset) -> SparkDataset:
        """Mean prediction for all fitted models.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predicted values.

        """
        assert self.models != [], "Should be fitted first."

        pred_col_prefix = "prediction"
        preds_dfs = [
            self.predict_single_fold(model, dataset).select(
                SparkDataset.ID_COLUMN,
                F.col(model.prediction_column).alias(f"{pred_col_prefix}_{i}")
            ) for i, model in enumerate(self.models)
        ]

        predict_sdf = self._average_predictions(dataset, preds_dfs, pred_col_prefix)

        preds_ds = dataset.empty()
        preds_ds = self._set_prediction(preds_ds, predict_sdf)

        return preds_ds

    def _average_predictions(self, preds_ds: SparkDataset, preds_dfs: List[SparkDataFrame], pred_col_prefix: str) -> SparkDataFrame:
        # TODO: SPARK-LAMA probably one may write a scala udf function to join multiple arrays/vectors into the one
        # TODO: reg and binary cases probably should be treated without arrays summation
        # we need counter here for EACH row, because for some models there may be no predictions
        # for some rows that means:
        # 1. we need to do left_outer instead of inner join (because the right frame may not contain all rows)
        # 2. we would like to find a mean prediction for each row, but the number of predictiosn may be variable,
        #    that is why we need a counter for each row
        # 3. this counter should depend on if there is None for the right row or not
        # 4. we also need to substitute None's of the right dataframe with empty arrays
        #    to provide uniformity for summing operations
        # 5. we also convert output from vector to an array to combine them
        counter_col_name = "counter"
        empty_pred = F.lit(np.zeros(self.n_classes))
        full_preds_df = preds_ds.data.select(
            SparkDataset.ID_COLUMN,
            F.lit(0).alias(counter_col_name),
            empty_pred
        )
        for i, pred_df in enumerate(preds_dfs):
            pred_col = f"{pred_col_prefix}_{i}"
            full_preds_df = (
                full_preds_df
                .join(pred_df, on=SparkDataset.ID_COLUMN, how="left_outer")
                .select(
                    full_preds_df[SparkDataset.ID_COLUMN],
                    F.when(F.col(pred_col).isNull(), empty_pred)
                        .otherwise(vector_to_array(F.col(pred_col))).alias(pred_col),
                    F.when(F.col(pred_col).isNull(), F.col(counter_col_name))
                        .otherwise(F.col(counter_col_name) + 1).alias(counter_col_name)
                )
                .select(
                    full_preds_df[SparkDataset.ID_COLUMN],
                    counter_col_name,
                    F.transform(
                        F.arrays_zip(full_preds_df[pred_col_prefix], f"{pred_col_prefix}_{i}"),
                        lambda x, y: x + y
                    ).alias(pred_col_prefix),
                )
            )

        full_preds_df = full_preds_df.select(
            SparkDataset.ID_COLUMN,
            array_to_vector(F.transform(pred_col_prefix, lambda x: x / F.col("counter"))).alias(pred_col_prefix)
        )

        return full_preds_df

    def _set_prediction(self, dataset: SparkDataset,  preds: SparkDataFrame) -> SparkDataset:
        """Insert predictions to dataset with. Inplace transformation.

        Args:
            dataset: Dataset to transform.
            preds: A spark dataframe  with predicted values.

        Returns:
            Transformed dataset.

        """
        prefix = f"{self._name}_prediction"
        prob = self.task.name in ["binary", "multiclass"]
        role = NumericVectorOrArrayRole(size=self.n_classes,
                                        element_col_name_template=prefix + "_{}",
                                        dtype=np.float32,
                                        force_input=True,
                                        prob=prob)
        dataset.set_data(preds, [prefix], role)

        return dataset

    @staticmethod
    def _make_sdf_with_target(train: SparkDataset) -> SparkDataFrame:
        """ Adds target column to the train data frame"""
        sdf = train.data
        t_sdf = train.target.select(SparkDataset.ID_COLUMN, train.target_column)
        if train.target_column not in train.data.columns:
            sdf = sdf.join(t_sdf, SparkDataset.ID_COLUMN).drop(t_sdf[SparkDataset.ID_COLUMN])
        return sdf
