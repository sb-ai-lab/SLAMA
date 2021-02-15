""" Uplift meta-models """

import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from log_calls import record_history
from pandas import DataFrame

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.automl.base import AutoML
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import UpliftIterator

from .utils import create_linear_automl, _get_target_role, _get_treatment_role


@record_history(enabled=False)
class MetaLearner(metaclass=ABCMeta):
    """Base class for uplift meta-learner"""
    @abstractmethod
    def fit(self, train_data: DataFrame, roles: Dict):
        pass

    @abstractmethod
    def predict(self, train_data: DataFrame) -> Tuple[np.ndarray, ...]:
        pass

    def _get_default_learner(self, task: Task):
        return create_linear_automl(task)


@record_history(enabled=False)
class TLearner(MetaLearner):
    """TLearner

    `TLearner` - is an 'meta' model which uses a two separated models.

    Each model is trained on it's own group (treatment/control).

    The 'meta' model prediction is a substraction predictions of 'treatment' model and 'control' model.

    """

    def __init__(self,
                 treatment_learner: Optional[AutoML] = None,
                 control_learner: Optional[AutoML] = None,
                 base_task: Optional[Task] = None):
        """
        Args:
            treatment_learner: AutoML model, if `None` then will be used model by default
            control_learner: AutoML model, if `None` then will be used model by default
            base_task: task

        """
        assert any(x is not None for x in [treatment_learner, control_learner, base_task]), (
               'Must specify any of learners or "base_task"')

        if base_task is None and (treatment_learner is None or control_learner is None):
            if treatment_learner is not None:
                base_task = treatment_learner.reader.task
            elif control_learner is not None:
                base_task = control_learner.reader.task

        self.treatment_learner = treatment_learner if treatment_learner is not None else self._get_default_learner(base_task)
        self.control_learner = control_learner if control_learner is not None else self._get_default_learner(base_task)

    def fit(self, train_data: DataFrame, roles: Dict):
        """Fit meta-learner

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        treatment_role, treatment_col = _get_treatment_role(roles)

        new_roles = copy.deepcopy(roles)
        new_roles.pop(treatment_role)

        control_train_data = train_data[train_data[treatment_col] == 0]
        treatment_train_data = train_data[train_data[treatment_col] == 1]

        control_train_data.drop(treatment_col, axis=1, inplace=True)
        treatment_train_data.drop(treatment_col, axis=1, inplace=True)

        self.treatment_learner.fit_predict(treatment_train_data, new_roles)
        self.control_learner.fit_predict(control_train_data, new_roles)

    def predict(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict treatment effects

        Args:
            data: Dataset to perform inference.

        Returns:
            treatment_effect: Predictions of treatment effects
            effect_w_interaction: Predictions of base task values on treated-group
            effect_wo_interaction: Predictions of base task values on control-group

        """
        treatment_pred = self.treatment_learner.predict(data).data.ravel()
        control_pred = self.control_learner.predict(data).data.ravel()

        uplift = treatment_pred - control_pred

        return uplift, treatment_pred, control_pred


@record_history(enabled=False)
class T2Learner(MetaLearner):
    """T2Learner

    `T2Learner` - is a modification of `TLearner` model.

    Difference of original approach is a training scheme (`UpliftIterator`).

    To train the base task of the model (treatment/control), use both parts of datasets (treatment/control) with
    additional feature `treatment_flag`.

    Each model is tuned on corresponding dataset.

    """

    def __init__(self,
                 treatment_learner: Optional[AutoML] = None,
                 control_learner: Optional[AutoML] = None,
                 n_uplift_iterator_folds: int = 5,
                 base_task: Optional[Task] = None):
        """
        Args:
            treatment_learner: AutoML model, if `None` then will be used model by default
            control_learner: AutoML model, if `None` then will be used model by default
            base_task: task
        """
        assert any(x is not None for x in [treatment_learner, control_learner, base_task]), (
               'Must specify any of learners or "base_task"')

        self._n_uplift_iterator_folds = n_uplift_iterator_folds

        if base_task is None:
            if treatment_learner is not None:
                if isinstance(treatment_learner, TabularAutoML):
                    base_task = treatment_learner.task
                else:
                    base_task = treatment_learner.reader.task
            else:
                if isinstance(control_learner, TabularAutoML):
                    base_task = control_learner
                else:
                    base_task = control_learner.reader.task

        self.base_task = base_task

        self.treatment_learner = treatment_learner if treatment_learner is not None else self._get_default_learner(base_task)
        self.control_learner = control_learner if control_learner is not None else self._get_default_learner(base_task)

    def fit(self, train_data: DataFrame, roles: Dict):
        """Fit meta-learner

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        treatment_role, treatment_col = _get_treatment_role(roles)
        _, target_col = _get_target_role(roles)
        self._treatment_col = treatment_col

        new_roles = copy.deepcopy(roles)
        new_roles.pop(treatment_role)

        train_data_c = train_data.copy()
        treatment_values = train_data_c[treatment_col].values
        target_values = train_data[target_col].values

        treatment_iterator = UpliftIterator(treatment_values, target_values, True,
                                            self.base_task, self._n_uplift_iterator_folds)
        self.treatment_learner.fit_predict(train_data_c, new_roles, cv_iter=treatment_iterator)

        control_iterator = UpliftIterator(treatment_values, target_values, False,
                                          self.base_task, self._n_uplift_iterator_folds)
        self.control_learner.fit_predict(train_data_c, new_roles, cv_iter=control_iterator)

    def predict(self, data: DataFrame):
        """Predict treatment effects

        Args:
            data: Dataset to perform inference.

        Returns:
            treatment_effect: Predictions of treatment effects
            effect_w_interaction: Predictions of base task values on treated-group
            effect_wo_interaction: Predictions of base task values on control-group

        """
        data_с = data.copy()
        data_с[self._treatment_col] = True
        treatment_pred = self.treatment_learner.predict(data_с).data.ravel()
        data_с[self._treatment_col] = False
        control_pred = self.control_learner.predict(data_с).data.ravel()

        uplift = treatment_pred - control_pred

        return uplift, treatment_pred, control_pred


@record_history(enabled=False)
class XLearner(MetaLearner):
    """XLearner

    XLearner - is a 'meta' model which use approach from `TLearner` `meta` model.

    The learning algorithm:

    Step #1 (Propensity score): Train the model to distinguish between the target and control groups.
    Step #2 (Outcome): Train two models on treatment/control group to predict base task, named this `outcome` model.
    Step #3 (Effect): Train models to predict difference between true outcome of treatment part dataset and prediction of
        `outcome-control` model (step 1) on treatment group dataset, and same for outcome, but with negative sign,
        named this `effect` model.

    Final prediction of `XLearner` is weighted sum of `effect` models (treatment/control), where weights is propensity score.

    """

    def __init__(self,
                 outcome_learners: Optional[Sequence[AutoML]] = None,
                 effect_learners: Optional[Sequence[AutoML]] = None,
                 propensity_learner: Optional[AutoML] = None,
                 base_task: Optional[Task] = None):
        """
        Args:
            outcome_learners: Models predict `outcome` (base task) for each group (treatment/control),
                base task can be classification or regression task.
                It can be: two models, one model or nothing.
                If there is one model, then it will used for both groups.
                If `None` then will be used model by default.
            effect_learners:  Models predict treatment effect. (task must be 'reg')
                It can be: two models, one model or nothing.
                If there is one model, then it will used for both groups.
                If `None` then will be used model by default.
            propensity_learner: Model predicts treatment group membership,
                If `None` then will be used model by default
            base_task: Task - 'binary' or 'reg'

        """
        if (outcome_learners is None or len(outcome_learners) == 0) and base_task is None:
            raise RuntimeError('Must specify any of learners or "base_task"')

        self.learners: Dict[str, Union[Dict[str, AutoML], AutoML]] = {'outcome': {}, 'effect': {}}
        if propensity_learner is None:
            self.learners['propensity'] = self._get_default_learner(Task('binary'))
        else:
            self.learners['propensity'] = propensity_learner

        if outcome_learners is None or len(outcome_learners) == 0:
            self.learners['outcome']['control'] = self._get_default_learner(base_task)
            self.learners['outcome']['treatment'] = self._get_default_learner(base_task)
        elif len(outcome_learners) == 1:
            self.learners['outcome']['control'] = outcome_learners[0]
            self.learners['outcome']['treatment'] = copy.deepcopy(outcome_learners[0])
        elif len(outcome_learners) == 2:
            self.learners['outcome']['control'] = outcome_learners[0]
            self.learners['outcome']['treatment'] = outcome_learners[1]
        else:
            raise RuntimeError('The number of "outcome_learners" must be 0/1/2')

        if effect_learners is None or len(effect_learners) == 0:
            self.learners['effect']['control'] = self._get_default_learner(Task('reg'))
            self.learners['effect']['treatment'] = self._get_default_learner(Task('reg'))
        elif len(effect_learners) == 1:
            self.learners['effect']['control'] = effect_learners[0]
            self.learners['effect']['treatment'] = copy.deepcopy(effect_learners[0])
        elif len(effect_learners) == 2:
            self.learners['effect']['control'] = effect_learners[0]
            self.learners['effect']['treatment'] = effect_learners[1]
        else:
            raise RuntimeError('The number of "effect_learners" must be 0/1/2')

    def fit(self, train_data: DataFrame, roles: Dict):
        """Fit meta-learner

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        self._fit_propensity_learner(train_data, roles)
        self._fit_outcome_learners(train_data, roles)
        self._fit_effect_learners(train_data, roles)

    def _fit_propensity_learner(self, train_data: DataFrame, roles: Dict):
        """Fit propensity score

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        propensity_roles = copy.deepcopy(roles)

        target_role, target_col = _get_target_role(roles)
        propensity_roles.pop(target_role)

        treatment_role, treatment_col = _get_treatment_role(roles)
        propensity_roles.pop(treatment_role)
        propensity_roles['target'] = treatment_col

        train_cp = train_data.copy()
        train_cp.drop(target_col, axis=1, inplace=True)

        self.learners['propensity'].fit_predict(train_cp, propensity_roles)

    def _fit_outcome_learners(self, train_data: DataFrame, roles: Dict):
        """Fit outcome

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        treatment_role, treatment_col = _get_treatment_role(roles)
        outcome_roles = copy.deepcopy(roles)
        outcome_roles.pop(treatment_role)

        for group_name, outcome_learner in self.learners['outcome'].items():
            group = 1 if group_name == 'treatment' else 0

            train_data_outcome = train_data[train_data[treatment_col] == group].copy()
            train_data_outcome.drop(treatment_col, axis=1, inplace=True)

            outcome_learner.fit_predict(train_data_outcome, outcome_roles)

    def _fit_effect_learners(self, train_data: DataFrame, roles: Dict):
        """Fit treatment effects

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        treatment_role, treatment_col = _get_treatment_role(roles)
        _, target_col = _get_target_role(roles)

        effect_roles: Dict = copy.deepcopy(roles)
        effect_roles.pop(treatment_role)

        for group_name, effect_learner in self.learners['effect'].items():
            group = 1 if group_name == 'treatment' else 0
            opposite_group_name = 'treatment' if group_name == 'control' else 'control'

            train_data_effect = train_data[train_data[treatment_col] == group].copy()
            train_data_effect.drop(treatment_col, axis=1, inplace=True)

            outcome_pred = self.learners['outcome'][opposite_group_name].predict(train_data_effect).data.ravel()
            train_data_effect[target_col] = train_data_effect[target_col] - outcome_pred

            if group_name == 'control':
                train_data_effect[target_col] *= -1

            effect_learner.fit_predict(train_data_effect, effect_roles)

    def predict(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict treatment effects

        Args:
            data: Dataset to perform inference.

        Returns:
            treatment_effect: Predictions of treatment effects
            effect_w_interaction: Predictions of base task values on treated-group
            effect_wo_interaction: Predictions of base task values on control-group

        """
        outcome_control_pred = self.learners['outcome']['control'].predict(data).data.ravel()
        outcome_treatment_pred = self.learners['outcome']['treatment'].predict(data).data.ravel()

        propensity_score = self.learners['propensity'].predict(data).data.ravel()
        uplift_control_pred = self.learners['effect']['control'].predict(data).data.ravel()
        uplift_treatment_pred = self.learners['effect']['treatment'].predict(data).data.ravel()
        uplift = propensity_score * uplift_treatment_pred + (1.0 - propensity_score) * uplift_control_pred

        return uplift, outcome_treatment_pred, outcome_control_pred


@record_history(enabled=False)
class RLearner(MetaLearner):
    """RLearner

    m(x) - the conditional mean outcome
    e(x) - the propensity score
    tau(x) - the treatment effect

    .. math::
        \tau(\cdot) = argmin_{\tau} \sum_{i} \Big[ (Y_i - m(X_i)) + (W_i - e(X_i))\tau(X_i) \Big]^2

    """
    def __init__(self,
                 propensity_learner: Optional[AutoML] = None,
                 mean_outcome_learner: Optional[AutoML] = None,
                 effect_learner: Optional[AutoML] = None,
                 base_task: Optional[Task] = Task('binary')):
        """
        Args:
            propensity_learner: AutoML model, if `None` then will be used model by default (task must be 'binary')
            mean_outcome_learner: AutoML model, if `None` then will be used model by default
            effect_learner: AutoML model, if `None` then will be used model by default (task must be 'reg')
            base_task: task

        """
        assert propensity_learner is not None and propensity_learner.reader.task.name == 'binary',\
            "Task of effect_learner must be 'reg'"
        assert not (mean_outcome_learner is None and base_task is None), "Must specify 'mean_outcome_learner' or base_task"
        assert effect_learner is not None and effect_learner.reader.task.name == 'reg', "Task of effect_learner must be 'reg'"

        self.propensity_learner: AutoML
        self.mean_outcome_learner: AutoML
        self.effect_learner: AutoML
        self.base_task = base_task

        if propensity_learner is None:
            self.propensity_learner = TabularAutoML(task='binary')
        else:
            self.propensity_learner = propensity_learner

        if mean_outcome_learner is not None:
            self.mean_outcome_learner = mean_outcome_learner
        elif base_task is not None:
            self.mean_outcome_learner = TabularAutoML(task=base_task)

        if effect_learner is None:
            self.effect_learner = TabularAutoML(task='reg')
        else:
            self.effect_learner = effect_learner

    def fit(self, train_data: DataFrame, roles: Dict):
        """Fit meta-learner

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        propensity_pred = self._fit_predict_propensity_learner(train_data, roles)
        mean_outcome_pred = self._fit_predict_mean_outcome_learner(train_data, roles)
        self._fit_effect_learner(train_data, roles, propensity_pred, mean_outcome_pred)

    def predict(self, data: Any) -> Tuple[np.ndarray, None, None]:
        """Predict treatment effects

        Args:
            data: Dataset to perform inference.

        Returns:
            treatment_effect: Predictions of treatment effects
            None: Plug
            None: Plug

        """
        return self.effect_learner.predict(data).data.ravel(), None, None

    def _fit_predict_propensity_learner(self, train_data: DataFrame, roles: Dict):
        """Fit propensity score

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        propensity_roles = copy.deepcopy(roles)

        target_role, target_col = _get_target_role(roles)
        propensity_roles.pop(target_role)

        treatment_role, treatment_col = _get_treatment_role(roles)
        propensity_roles.pop(treatment_role)
        propensity_roles['target'] = treatment_col

        train_cp = train_data.copy()
        train_cp.drop(target_col, axis=1, inplace=True)

        propensity_pred = self.propensity_learner.fit_predict(train_cp, propensity_roles).data.ravel()

        return propensity_pred

    def _fit_predict_mean_outcome_learner(self, train_data: DataFrame, roles: Dict):
        """Fit mean outcome

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        outcome_roles = copy.deepcopy(roles)

        target_role, target_col = _get_target_role(roles)

        treatment_role, treatment_col = _get_treatment_role(roles)
        outcome_roles.pop(treatment_role)

        train_cp = train_data.copy()
        train_cp.drop(treatment_col, axis=1, inplace=True)

        mean_outcome_pred = self.mean_outcome_learner.fit_predict(train_cp, outcome_roles).data.ravel()

        return mean_outcome_pred

    def _fit_effect_learner(self, train_data: DataFrame, roles: Dict, propensity_pred: np.ndarray,
                            mean_outcome_pred: np.ndarray):
        """Fit treatment effects

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles
            propensity_pred: oof-prediction of propensity_learner
            mean_outcome_pred: oof-prediction of mean_outcome_learner

        """
        epsi = 10 ** -5
        effect_roles = copy.deepcopy(roles)

        target_role, target_col = _get_target_role(roles)
        train_target = train_data[target_col]

        treatment_role, treatment_col = _get_treatment_role(roles)
        train_treatment = train_data[treatment_col]
        effect_roles.pop(treatment_role)

        weights = train_treatment - propensity_pred + epsi

        train_cp = train_data.copy()
        train_cp.drop(treatment_col, axis=1, inplace=True)
        train_cp[target_col] = (train_target - mean_outcome_pred) / weights
        train_cp['__WEIGHTS__'] = weights ** 2

        effect_roles['weights'] = '__WEIGHTS__'

        self.effect_learner.fit_predict(train_cp, effect_roles)
