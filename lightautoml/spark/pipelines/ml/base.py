"""Base classes for MLPipeline."""

from typing import List, cast

from lightautoml.validation.base import TrainValidIterator
from ...dataset.base import LAMLDataset, SparkDataset
from ....ml_algo.utils import tune_and_fit_predict
from ....pipelines.ml.base import MLPipeline as LAMAMLPipeline


class SparkMLPipelineMixin(LAMAMLPipeline):
    def fit_predict(self, train_valid: TrainValidIterator) -> LAMLDataset:
        """Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: Dataset iterator.

        Returns:
            Dataset with predictions of all models.

        """
        self.ml_algos = []

        with cast(SparkDataset, train_valid.train).applying_temporary_caching():
            # train and apply pre selection
            train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        with cast(SparkDataset, train_valid.train).applying_temporary_caching():
            train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

        # train and apply post selection
        with cast(SparkDataset, train_valid.train).applying_temporary_caching():
            train_valid = train_valid.apply_selector(self.post_selection)

        predictions = []

        train_ds = cast(SparkDataset, train_valid.train)
        with train_ds.applying_temporary_caching():
            for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc):
                ml_algo, preds = tune_and_fit_predict(ml_algo, param_tuner, train_valid, force_calc)
                if ml_algo is not None:
                    self.ml_algos.append(ml_algo)
                    preds = cast(SparkDataset, preds)
                    predictions.append(preds)

            assert (
                len(predictions) > 0
            ), "Pipeline finished with 0 models for some reason.\nProbably one or more models failed"

            predictions = SparkDataset.concatenate(predictions)
            predictions.cache_and_materialize()
        # TODO: clean anything that can be cached in tune_and_fit_predict

        del self._ml_algos
        return predictions

    def predict(self, dataset: LAMLDataset) -> LAMLDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predictions of all trained models.

        """
        dataset = self.pre_selection.select(dataset)
        dataset = self.features_pipeline.transform(dataset)
        dataset = self.post_selection.select(dataset)

        predictions: List[SparkDataset] = []

        dataset = cast(SparkDataset, dataset)

        # TODO: SPARK-LAMA same problem with caching - we don't know when to uncache
        # we should uncache only after the whole AutoML workflow is materialized
        dataset.cache()

        for model in self.ml_algos:
            pred = cast(SparkDataset, model.predict(dataset))
            predictions.append(pred)

        result = SparkDataset.concatenate(predictions)

        return result
