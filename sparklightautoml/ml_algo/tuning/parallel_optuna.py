import logging
from copy import deepcopy
from typing import Optional, Tuple, Callable

import optuna
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.validation.base import HoldoutIterator

from sparklightautoml.computations.manager import _SlotInitiatedTVIter, SlotAllocator, \
    PoolType, \
    SequentialComputationsManager, ComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)


class ParallelOptunaTuner(OptunaTuner):
    def __init__(self,
                 timeout: Optional[int] = 1000,
                 n_trials: Optional[int] = 100,
                 direction: Optional[str] = "maximize",
                 fit_on_holdout: bool = True,
                 random_state: int = 42,
                 parallelism: int = 1):
        super().__init__(timeout, n_trials, direction, fit_on_holdout, random_state)
        self._parallelism = parallelism

    def fit(self, ml_algo: SparkTabularMLAlgo, train_valid_iterator: Optional[SparkBaseTrainValidIterator] = None) \
            -> Tuple[Optional[SparkTabularMLAlgo], Optional[SparkDataset]]:
        """Tune model.

               Args:
                   ml_algo: Algo that is tuned.
                   train_valid_iterator: Classic cv-iterator.

               Returns:
                   Tuple (None, None) if an optuna exception raised
                   or ``fit_on_holdout=True`` and ``train_valid_iterator`` is
                   not :class:`~lightautoml.validation.base.HoldoutIterator`.
                   Tuple (MlALgo, preds_ds) otherwise.

               """
        assert not ml_algo.is_fitted, "Fitted algo cannot be tuned."

        estimated_tuning_time = ml_algo.timer.estimate_tuner_time(len(train_valid_iterator))
        if estimated_tuning_time:
            estimated_tuning_time = max(estimated_tuning_time, 1)
            self._upd_timeout(estimated_tuning_time)

        logger.info(
            f"Start hyperparameters optimization for \x1b[1m{ml_algo._name}\x1b[0m ... Time budget is {self.timeout:.2f} secs"
        )

        metric_name = train_valid_iterator.train.task.get_dataset_metric().name
        ml_algo = deepcopy(ml_algo)

        flg_new_iterator = False
        if self._fit_on_holdout and type(train_valid_iterator) != HoldoutIterator:
            train_valid_iterator = train_valid_iterator.convert_to_holdout_iterator()
            flg_new_iterator = True

        def update_trial_time(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            """Callback for number of iteration with time cut-off.

            Args:
                study: Optuna study object.
                trial: Optuna trial object.

            """
            ml_algo.mean_trial_time = study.trials_dataframe()["duration"].mean().total_seconds()
            self.estimated_n_trials = min(self.n_trials, self.timeout // ml_algo.mean_trial_time)

            logger.info3(
                f"\x1b[1mTrial {len(study.trials)}\x1b[0m with hyperparameters {trial.params} scored {trial.value} in {trial.duration}"
            )

        try:

            self._optimize(ml_algo, train_valid_iterator, update_trial_time)

            # need to update best params here
            self._best_params = self.study.best_params
            ml_algo.params = self._best_params

            logger.info(f"Hyperparameters optimization for \x1b[1m{ml_algo._name}\x1b[0m completed")
            logger.info2(
                f"The set of hyperparameters \x1b[1m{self._best_params}\x1b[0m\n achieve {self.study.best_value:.4f} {metric_name}"
            )

            if flg_new_iterator:
                # if tuner was fitted on holdout set we dont need to save train results
                return None, None

            preds_ds = ml_algo.fit_predict(train_valid_iterator)

            return ml_algo, preds_ds
        except optuna.exceptions.OptunaError:
            return None, None

    def _optimize(self,
                  ml_algo: SparkTabularMLAlgo,
                  train_valid_iterator: SparkBaseTrainValidIterator,
                  update_trial_time: Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]):

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)

        ml_algo = deepcopy(ml_algo)
        ml_algo.computations_manager = SequentialComputationsManager()

        self.study.optimize(
            func=self._get_objective(
                ml_algo=ml_algo,
                estimated_n_trials=self.estimated_n_trials,
                train_valid_iterator=train_valid_iterator,
            ),
            n_jobs=self._parallelism,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[update_trial_time],
        )


class SlotBasedParallelOptunaTuner(ParallelOptunaTuner):
    def __init__(self,
                 computations_manager: ComputationsManager,
                 timeout: Optional[int] = 1000,
                 n_trials: Optional[int] = 100,
                 direction: Optional[str] = "maximize",
                 fit_on_holdout: bool = True,
                 random_state: int = 42,
                 parallelism: int = 1):
        super().__init__(timeout, n_trials, direction, fit_on_holdout, random_state, parallelism)
        self._computations_manager = computations_manager

        assert self._computations_manager.can_support_slots, "The computation manager should support slots allocation!"

    def _optimize(self, ml_algo: SparkTabularMLAlgo, train_valid_iterator: SparkBaseTrainValidIterator,
                  update_trial_time: Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]):
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)

        with self._computations_manager.slots(train_valid_iterator.train,
                                              parallelism=self._parallelism, pool_type=PoolType.job) as allocator:
            allocator: SlotAllocator = allocator
            ml_algo = deepcopy(ml_algo)
            ml_algo.computations_manager = SequentialComputationsManager()

            self.study.optimize(
                func=self._get_objective(
                    ml_algo=ml_algo,
                    estimated_n_trials=self.estimated_n_trials,
                    train_valid_iterator=_SlotInitiatedTVIter(allocator, train_valid_iterator),
                ),
                n_jobs=self._parallelism,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[update_trial_time]
            )
