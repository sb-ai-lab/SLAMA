import logging

from copy import deepcopy
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import optuna

from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.ml_algo.tuning.optuna import TunableAlgo
from lightautoml.validation.base import HoldoutIterator

from sparklightautoml.computations.base import ComputationsSession
from sparklightautoml.computations.base import ComputationsSettings
from sparklightautoml.computations.builder import build_computations_manager
from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.computations.utils import deecopy_tviter_without_dataset
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.validation.base import SparkBaseTrainValidIterator


logger = logging.getLogger(__name__)


class ParallelOptunaTuner(OptunaTuner):
    def __init__(
        self,
        timeout: Optional[int] = 1000,
        n_trials: Optional[int] = 100,
        direction: Optional[str] = "maximize",
        fit_on_holdout: bool = True,
        random_state: int = 42,
        parallelism: int = 1,
        computations_manager: Optional[ComputationsSettings] = None,
    ):
        super().__init__(timeout, n_trials, direction, fit_on_holdout, random_state)
        self._parallelism = parallelism
        self._computations_manager = build_computations_manager(computations_settings=computations_manager)
        self._session: Optional[ComputationsSession] = None

    def fit(
        self, ml_algo: SparkTabularMLAlgo, train_valid_iterator: Optional[SparkBaseTrainValidIterator] = None
    ) -> Tuple[Optional[SparkTabularMLAlgo], Optional[SparkDataset]]:
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
        except:
            logger.error("Error during parameters optimization", exc_info=True)
            raise

    def _optimize(
        self,
        ml_algo: SparkTabularMLAlgo,
        train_valid_iterator: SparkBaseTrainValidIterator,
        update_trial_time: Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None],
    ):
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)

        # prepare correct ml_algo to run with optuna
        cm = ml_algo.computations_manager
        trial_ml_algo = deepcopy(ml_algo)
        ml_algo.computations_manager = cm
        trial_ml_algo.persist_output_dataset = False

        with self._computations_manager.session(train_valid_iterator.train) as self._session:
            # _get objective will use self._session
            self.study.optimize(
                func=self._get_objective(
                    ml_algo=trial_ml_algo,
                    estimated_n_trials=self.estimated_n_trials,
                    train_valid_iterator=train_valid_iterator,
                ),
                n_jobs=self._parallelism,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[update_trial_time],
            )

        self._session = None

    def _get_objective(
        self, ml_algo: TunableAlgo, estimated_n_trials: int, train_valid_iterator: SparkBaseTrainValidIterator
    ) -> Callable[[optuna.trial.Trial], Union[float, int]]:
        assert isinstance(ml_algo, SparkTabularMLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            with self._session.allocate() as slot:
                assert slot.dataset is not None
                _ml_algo = deepcopy(ml_algo)
                _ml_algo.computations_manager = SequentialComputationsManager(
                    num_tasks=slot.num_tasks, num_threads_per_executor=slot.num_threads_per_executor
                )
                tv_iter = deecopy_tviter_without_dataset(train_valid_iterator)
                tv_iter.train = slot.dataset

                optimization_search_space = _ml_algo.optimization_search_space

                if not optimization_search_space:
                    optimization_search_space = _ml_algo._get_default_search_spaces(
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                        estimated_n_trials=estimated_n_trials,
                    )

                if callable(optimization_search_space):
                    _ml_algo.params = optimization_search_space(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                    )
                else:
                    _ml_algo.params = self._sample(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                    )

                logger.debug(f"Sampled ml_algo params: {_ml_algo.params}")

                output_dataset = _ml_algo.fit_predict(train_valid_iterator=tv_iter)

                return _ml_algo.score(output_dataset)

        return objective
