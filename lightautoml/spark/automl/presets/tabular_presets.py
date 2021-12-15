import logging
import os
from copy import deepcopy, copy
from typing import Optional, Sequence, Iterable, cast, Union, Tuple, Callable

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from lightautoml.automl.presets.base import AutoMLPreset, upd_params
from lightautoml.dataset.base import RolesDict, LAMLDataset
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.pipelines.selection.importance_based import ModelBasedImportanceEstimator, ImportanceCutoffSelector
from lightautoml.reader.tabular_batch_generator import ReadableToDf, read_data
from lightautoml.spark.automl.blend import WeightedBlender
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBAdvancedPipeline
from lightautoml.spark.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.spark.pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.tasks import Task

logger = logging.getLogger(__name__)

# Either path/full url, or pyspark.sql.DataFrame, or dict with data
ReadableIntoSparkDf = Union[str, SparkDataFrame, dict, pd.DataFrame]


class TabularAutoML(AutoMLPreset):
    _default_config_path = "tabular_config.yml"

    # set initial runtime rate guess for first level models
    _time_scores = {
        "lgb": 1,
        "lgb_tuned": 3,
        "linear_l2": 0.7,
        "cb": 2,
        "cb_tuned": 6,
    }

    def __init__(
            self,
            spark: SparkSession,
            task: Task,
            timeout: int = 3600,
            memory_limit: int = 16,
            cpu_limit: int = 4,
            gpu_ids: Optional[str] = "all",
            timing_params: Optional[dict] = None,
            config_path: Optional[str] = None,
            general_params: Optional[dict] = None,
            reader_params: Optional[dict] = None,
            read_csv_params: Optional[dict] = None,
            nested_cv_params: Optional[dict] = None,
            tuning_params: Optional[dict] = None,
            selection_params: Optional[dict] = None,
            lgb_params: Optional[dict] = None,
            cb_params: Optional[dict] = None,
            linear_l2_params: Optional[dict] = None,
            gbm_pipeline_params: Optional[dict] = None,
            linear_pipeline_params: Optional[dict] = None,
    ):
        super().__init__(task, timeout, memory_limit, cpu_limit, gpu_ids, timing_params, config_path)

        logger.info("I'm here")

        self._spark = spark
        # upd manual params
        for name, param in zip(
                [
                    "general_params",
                    "reader_params",
                    "read_csv_params",
                    "nested_cv_params",
                    "tuning_params",
                    "selection_params",
                    "lgb_params",
                    "cb_params",
                    "linear_l2_params",
                    "gbm_pipeline_params",
                    "linear_pipeline_params",
                ],
                [
                    general_params,
                    reader_params,
                    read_csv_params,
                    nested_cv_params,
                    tuning_params,
                    selection_params,
                    lgb_params,
                    cb_params,
                    linear_l2_params,
                    gbm_pipeline_params,
                    linear_pipeline_params,
                ],
        ):
            if param is None:
                param = {}
            self.__dict__[name] = upd_params(self.__dict__[name], param)

    # TODO: SPARK-LAMA rewrite in the descdent
    def infer_auto_params(self, train_data: SparkDataFrame, multilevel_avail: bool = False):
        # infer optuna tuning iteration based on dataframe len
        if self.tuning_params["max_tuning_iter"] == "auto":
            if not train_data.is_cached:
                self.tuning_params["max_tuning_iter"] = 5

            length = train_data.count()

            if length < 10000:
                self.tuning_params["max_tuning_iter"] = 100
            elif length < 30000:
                self.tuning_params["max_tuning_iter"] = 50
            elif length < 100000:
                self.tuning_params["max_tuning_iter"] = 10
            else:
                self.tuning_params["max_tuning_iter"] = 5

        if self.general_params["use_algos"] == "auto":
            # TODO: More rules and add cases
            self.general_params["use_algos"] = [["lgb", "lgb_tuned", "linear_l2", "cb", "cb_tuned"]]
            if self.task.name == "multiclass" and multilevel_avail:
                self.general_params["use_algos"].append(["linear_l2", "lgb"])

        if not self.general_params["nested_cv"]:
            self.nested_cv_params["cv"] = 1


    def get_time_score(self, n_level: int, model_type: str, nested: Optional[bool] = None):

        if nested is None:
            nested = self.general_params["nested_cv"]

        score = self._time_scores[model_type]

        mult = 1
        if nested:
            if self.nested_cv_params["n_folds"] is not None:
                mult = self.nested_cv_params["n_folds"]
            else:
                mult = self.nested_cv_params["cv"]

        if n_level > 1:
            mult *= 0.8 if self.general_params["skip_conn"] else 0.1

        score = score * mult

        # lower score for catboost on gpu
        if model_type in ["cb", "cb_tuned"] and self.cb_params["default_params"]["task_type"] == "GPU":
            score *= 0.5
        return score

    # TODO: SPARK-LAMA rewrite in the descdent
    def get_selector(self, n_level: Optional[int] = 1) -> SelectionPipeline:
        selection_params = self.selection_params
        # lgb_params
        lgb_params = deepcopy(self.lgb_params)
        lgb_params["default_params"] = {
            **lgb_params["default_params"],
            **{"feature_fraction": 1},
        }

        mode = selection_params["mode"]

        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            # timer will be useful to estimate time for next gbm runs
            time_score = self.get_time_score(n_level, "lgb", False)

            sel_timer_0 = self.timer.get_task_timer("lgb", time_score)
            selection_feats = LGBSimpleFeatures()

            selection_gbm = BoostLGBM(timer=sel_timer_0, **lgb_params)
            selection_gbm.set_prefix("Selector")

            if selection_params["importance_type"] == "permutation":
                # TODO: SPARK-LAMA add support for this selector
                raise NotImplementedError("Not supported yet")
                # importance = NpPermutationImportanceEstimator()
            else:
                importance = ModelBasedImportanceEstimator()

            pre_selector = ImportanceCutoffSelector(
                selection_feats,
                selection_gbm,
                importance,
                cutoff=selection_params["cutoff"],
                fit_on_holdout=selection_params["fit_on_holdout"],
            )
            if mode == 2:
                time_score = self.get_time_score(n_level, "lgb", False)

                sel_timer_1 = self.timer.get_task_timer("lgb", time_score)
                selection_feats = LGBSimpleFeatures()
                selection_gbm = BoostLGBM(timer=sel_timer_1, **lgb_params)
                selection_gbm.set_prefix("Selector")

                # TODO: SPARK-LAMA add support for this selector
                raise NotImplementedError("Not supported yet")
                # # TODO: Check about reusing permutation importance
                # importance = NpPermutationImportanceEstimator()
                #
                # extra_selector = NpIterativeFeatureSelector(
                #     selection_feats,
                #     selection_gbm,
                #     importance,
                #     feature_group_size=selection_params["feature_group_size"],
                #     max_features_cnt_in_result=selection_params["max_features_cnt_in_result"],
                # )
                #
                # pre_selector = ComposedSelector([pre_selector, extra_selector])

        return pre_selector

    # TODO: SPARK-LAMA rewrite in the descdent
    def get_linear(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None) -> NestedTabularMLPipeline:

        # linear model with l2
        time_score = self.get_time_score(n_level, "linear_l2")
        linear_l2_timer = self.timer.get_task_timer("reg_l2", time_score)
        linear_l2_model = LinearLBFGS(timer=linear_l2_timer, **self.linear_l2_params)
        linear_l2_feats = LinearFeatures(output_categories=True, **self.linear_pipeline_params)

        linear_l2_pipe = NestedTabularMLPipeline(
            [linear_l2_model],
            force_calc=True,
            pre_selection=pre_selector,
            features_pipeline=linear_l2_feats,
            **self.nested_cv_params
        )
        return linear_l2_pipe

    # TODO: SPARK-LAMA rewrite in the descdent
    def get_gbms(
            self,
            keys: Sequence[str],
            n_level: int = 1,
            pre_selector: Optional[SelectionPipeline] = None,
    ):

        gbm_feats = LGBAdvancedPipeline(**self.gbm_pipeline_params)

        ml_algos = []
        force_calc = []
        for key, force in zip(keys, [True, False, False, False]):
            tuned = "_tuned" in key
            algo_key = key.split("_")[0]
            time_score = self.get_time_score(n_level, key)
            gbm_timer = self.timer.get_task_timer(algo_key, time_score)
            if algo_key == "lgb":
                gbm_model = BoostLGBM(timer=gbm_timer, **self.lgb_params)
            elif algo_key == "cb":
                # TODO: SPARK-LAMA implement this later
                raise NotImplementedError("Not supported yet")
                # gbm_model = BoostCB(timer=gbm_timer, **self.cb_params)
            else:
                raise ValueError("Wrong algo key")

            if tuned:
                gbm_model.set_prefix("Tuned")
                gbm_tuner = OptunaTuner(
                    n_trials=self.tuning_params["max_tuning_iter"],
                    timeout=self.tuning_params["max_tuning_time"],
                    fit_on_holdout=self.tuning_params["fit_on_holdout"],
                )
                gbm_model = (gbm_model, gbm_tuner)
            ml_algos.append(gbm_model)
            force_calc.append(force)

        gbm_pipe = NestedTabularMLPipeline(
            ml_algos, force_calc, pre_selection=pre_selector, features_pipeline=gbm_feats, **self.nested_cv_params
        )

        return gbm_pipe

    # TODO: SPARK-LAMA correct it to rewrite only some submethods in the descendant
    def create_automl(self, **fit_args):
        """Create basic automl instance.

        Args:
            **fit_args: Contain all information needed for creating automl.

        """
        train_data = fit_args["train_data"]
        multilevel_avail = fit_args["valid_data"] is None and fit_args["cv_iter"] is None

        self.infer_auto_params(train_data, multilevel_avail)
        reader = SparkToSparkReader(task=self.task, **self.reader_params)

        pre_selector = self.get_selector()

        levels = []

        for n, names in enumerate(self.general_params["use_algos"]):
            lvl = []
            # regs
            if "linear_l2" in names:
                selector = None
                if "linear_l2" in self.selection_params["select_algos"] and (
                        self.general_params["skip_conn"] or n == 0
                ):
                    selector = pre_selector
                lvl.append(self.get_linear(n + 1, selector))

            gbm_models = [
                x for x in ["lgb", "lgb_tuned", "cb", "cb_tuned"] if
                x in names and x.split("_")[0] in self.task.losses
            ]

            if len(gbm_models) > 0:
                selector = None
                if "gbm" in self.selection_params["select_algos"] and (self.general_params["skip_conn"] or n == 0):
                    selector = pre_selector
                lvl.append(self.get_gbms(gbm_models, n + 1, selector))

            levels.append(lvl)

        # blend everything
        blender = WeightedBlender(max_nonzero_coef=self.general_params["weighted_blender_max_nonzero_coef"])

        # initialize
        self._initialize(
            reader,
            levels,
            skip_conn=self.general_params["skip_conn"],
            blender=blender,
            return_all_predictions=self.general_params["return_all_predictions"],
            timer=self.timer,
        )

    # TODO: SPARK-LAMA should be renamed into a more general way
    def _get_read_csv_params(self):
        try:
            cols_to_read = self.reader.used_features
            numeric_dtypes = {
                x: self.reader.roles[x].dtype for x in self.reader.roles if self.reader.roles[x].name == "Numeric"
            }
        except AttributeError:
            cols_to_read = []
            numeric_dtypes = {}
        # cols_to_read is empty if reader is not fitted
        if len(cols_to_read) == 0:
            cols_to_read = None

        read_csv_params = copy(self.read_csv_params)
        read_csv_params = {
            **read_csv_params,
            **{"usecols": cols_to_read, "dtype": numeric_dtypes},
        }

        return read_csv_params

    def _read_data(self,
                   data: ReadableIntoSparkDf,
                   features_names: Optional[Sequence[str]] = None,
                   read_csv_params: Optional[dict] = None) -> Tuple[SparkDataFrame, Optional[RolesDict]]:
        """Get :class:`~pandas.DataFrame` from different data formats.

          Note:
              Supported now data formats:

                  - Path to ``.csv``, ``.parquet``, ``.feather`` files.
                  - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
                    For example, ``{'data': X...}``. In this case,
                    roles are optional, but `train_features`
                    and `valid_features` required.
                  - :class:`pandas.DataFrame`.

          Args:
              data: Readable to DataFrame data.
              features_names: Optional features names if ``numpy.ndarray``.
              n_jobs: Number of processes to read file.
              read_csv_params: Params to read csv file.

          Returns:
              Tuple with read data and new roles mapping.

          """
        if read_csv_params is None:
            read_csv_params = {}

        if isinstance(data, SparkDataFrame):
            return data, None

        if isinstance(data, pd.DataFrame):
            # TODO SPARK-LAMA: Fix schema inference
            object_cols = [c for c, dtype in dict(data.dtypes).items() if str(dtype) == 'object']
            for c in object_cols:
                data[c] = data[c].where(pd.notnull(data[c]), None)
            return self._spark.createDataFrame(data), None

        # case - dict of array args passed
        if isinstance(data, dict):
            # TODO: SPARK-LAMA implement it later
            raise NotImplementedError("Not supported yet")
            # df = self._spark.createDataFrame(data=data["data"])
            # upd_roles = {}
            # for k in data:
            #     if k != "data":
            #         name = "__{0}__".format(k.upper())
            #         assert name not in df.columns, "Not supported feature name {0}".format(name)
            #         df[name] = data[k]
            #         upd_roles[k] = name
            # return df, upd_roles

        if isinstance(data, str):
            path: str = data
            if path.endswith(".feather"):
                # TODO: SPARK-LAMA implement it later
                raise NotImplementedError("Not supported yet")
                # # TODO: check about feather columns arg
                # data = pd.read_feather(data)
                # if read_csv_params["usecols"] is not None:
                #     data = data[read_csv_params["usecols"]]
                # return data, None
            if path.endswith(".parquet"):
                return self._spark.read.parquet(path), None
                # return pd.read_parquet(data, columns=read_csv_params["usecols"]), None
            if path.endswith(".csv"):
                return self._spark.read.csv(path, **read_csv_params), None
            else:
                raise ValueError(f"Unsupported data format: {os.path.splitext(path)[1]}")

        raise ValueError("Input data format is not supported")

    # TODO: SPARK-LAMA rewrite in the descdent
    def fit_predict(
            self,
            train_data: ReadableIntoSparkDf,
            roles: Optional[dict] = None,
            train_features: Optional[Sequence[str]] = None,
            cv_iter: Optional[Iterable] = None,
            valid_data: Optional[ReadableIntoSparkDf] = None,
            valid_features: Optional[Sequence[str]] = None,
            log_file: str = None,
            verbose: int = 0,
    ) -> SparkDataset:
        """Fit and get prediction on validation dataset.

        Almost same as :meth:`lightautoml.automl.base.AutoML.fit_predict`.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
              For example, ``{'data': X...}``. In this case,
              roles are optional, but `train_features`
              and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Optional features names, if can't
              be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example,
              :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features
              if cannot be inferred from `valid_data`.
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;
            log_file: Filename for writing logging messages. If log_file is specified,
            the messages will be saved in a the file. If the file exists, it will be overwritten.

        Returns:
            Dataset with predictions. Call ``.data`` to get predictions array.

        """
        # roles may be none in case of train data is set {'data': np.ndarray, 'target': np.ndarray ...}
        self.set_logfile(log_file)

        if roles is None:
            roles = {}
        read_csv_params = self._get_read_csv_params()
        train, upd_roles = self._read_data(train_data, train_features, read_csv_params)
        if upd_roles:
            roles = {**roles, **upd_roles}
        if valid_data is not None:
            data, _ = self._read_data(valid_data, valid_features, read_csv_params)

        oof_pred = super().fit_predict(train, roles=roles, cv_iter=cv_iter, valid_data=valid_data, verbose=verbose)

        return cast(SparkDataset, oof_pred)

    # TODO: SPARK-LAMA rewrite in the descdent
    def predict(
            self,
            data: ReadableIntoSparkDf,
            features_names: Optional[Sequence[str]] = None,
            return_all_predictions: Optional[bool] = None,
    ) -> SparkDataset:

        read_csv_params = self._get_read_csv_params()
        data, _ = self._read_data(data, features_names, read_csv_params)
        pred = super().predict(data, features_names, return_all_predictions)
        return cast(SparkDataset, pred)

    # TODO: SPARK-LAMA rewrite in the descdent (or reload read_data)
    def get_feature_scores(
            self,
            calc_method: str = "fast",
            data: Optional[ReadableToDf] = None,
            features_names: Optional[Sequence[str]] = None,
            silent: bool = True,
    ):
        if calc_method == "fast":
            for level in self.levels:
                for pipe in level:
                    fi = pipe.pre_selection.get_features_score()
                    if fi is not None:
                        used_feats = set(self.collect_used_feats())
                        fi = fi.reset_index()
                        fi.columns = ["Feature", "Importance"]
                        return fi[fi["Feature"].map(lambda x: x in used_feats)]

            else:
                if not silent:
                    logger.info2("No feature importances to show. Please use another calculation method")
                return None

        if calc_method != "accurate":
            if not silent:
                logger.info2(
                    "Unknown calc_method. "
                    + "Currently supported methods for feature importances calculation are 'fast' and 'accurate'."
                )
            return None

        if data is None:
            if not silent:
                logger.info2("Data parameter is not setup for accurate calculation method. Aborting...")
            return None

        read_csv_params = self._get_read_csv_params()
        data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)
        used_feats = self.collect_used_feats()

        # TODO: SPARK-LAMA implement it later
        raise NotImplementedError("Not supported yet")
        # fi = calc_feats_permutation_imps(
        #     self,
        #     used_feats,
        #     data,
        #     self.reader.target,
        #     self.task.get_dataset_metric(),
        #     silent=silent,
        # )
        # return fi

    def get_individual_pdp(
            self,
            test_data: ReadableToDf,
            feature_name: str,
            n_bins: Optional[int] = 30,
            top_n_categories: Optional[int] = 10,
            datetime_level: Optional[str] = "year",
    ):
        # TODO: SPARK-LAMA implement it later
        raise NotImplementedError("Not supported yet")
        # assert feature_name in self.reader._roles
        # assert datetime_level in ["year", "month", "dayofweek"]
        # test_i = test_data.copy()
        # # Numerical features
        # if self.reader._roles[feature_name].name == "Numeric":
        #     counts, bin_edges = np.histogram(test_data[feature_name].dropna(), bins=n_bins)
        #     grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        #     ys = []
        #     for i in tqdm(grid):
        #         test_i[feature_name] = i
        #         preds = self.predict(test_i).data
        #         ys.append(preds)
        # # Categorical features
        # if self.reader._roles[feature_name].name == "Category":
        #     feature_cnt = test_data[feature_name].value_counts()
        #     grid = list(feature_cnt.index.values[:top_n_categories])
        #     counts = list(feature_cnt.values[:top_n_categories])
        #     ys = []
        #     for i in tqdm(grid):
        #         test_i[feature_name] = i
        #         preds = self.predict(test_i).data
        #         ys.append(preds)
        #     if len(feature_cnt) > top_n_categories:
        #         freq_mapping = {feature_cnt.index[i]: i for i, _ in enumerate(feature_cnt)}
        #         # add "OTHER" class
        #         test_i = test_data.copy()
        #         # sample from other classes with the same distribution
        #         test_i[feature_name] = (
        #             test_i[feature_name][np.array([freq_mapping[k] for k in test_i[feature_name]]) > top_n_categories]
        #                 .sample(n=test_data.shape[0], replace=True)
        #                 .values
        #         )
        #         preds = self.predict(test_i).data
        #         grid.append("<OTHER>")
        #         ys.append(preds)
        #         counts.append(feature_cnt.values[top_n_categories:].sum())
        # # Datetime Features
        # if self.reader._roles[feature_name].name == "Datetime":
        #     test_data_read = self.reader.read(test_data)
        #     feature_datetime = pd.arrays.DatetimeArray(test_data_read._data[feature_name])
        #     if datetime_level == "year":
        #         grid = np.unique([i.year for i in feature_datetime])
        #     elif datetime_level == "month":
        #         grid = np.arange(1, 13)
        #     else:
        #         grid = np.arange(7)
        #     ys = []
        #     for i in tqdm(grid):
        #         test_i[feature_name] = change_datetime(feature_datetime, datetime_level, i)
        #         preds = self.predict(test_i).data
        #         ys.append(preds)
        #     counts = Counter([getattr(i, datetime_level) for i in feature_datetime])
        #     counts = [counts[i] for i in grid]
        # return grid, ys, counts

    def plot_pdp(
            self,
            test_data: ReadableToDf,
            feature_name: str,
            individual: Optional[bool] = False,
            n_bins: Optional[int] = 30,
            top_n_categories: Optional[int] = 10,
            top_n_classes: Optional[int] = 10,
            datetime_level: Optional[str] = "year",
    ):
        # TODO: SPARK-LAMA implement it later
        raise NotImplementedError("Not supported yet")
        # grid, ys, counts = self.get_individual_pdp(
        #     test_data=test_data,
        #     feature_name=feature_name,
        #     n_bins=n_bins,
        #     top_n_categories=top_n_categories,
        #     datetime_level=datetime_level,
        # )
        # plot_pdp_with_distribution(
        #     test_data,
        #     grid,
        #     ys,
        #     counts,
        #     self.reader,
        #     feature_name,
        #     individual,
        #     top_n_classes,
        #     datetime_level,
        # )

    def _concatenate_datasets(self, datasets: Sequence[LAMLDataset]) -> LAMLDataset:
        assert all(isinstance(ds, SparkDataset) for ds in datasets)
        sdss = [cast(SparkDataset, ds) for ds in datasets]
        return SparkDataset.concatenate(sdss)

    def _create_validation_iterator(self, train: LAMLDataset, valid: Optional[LAMLDataset], n_folds: Optional[int],
                                    cv_iter: Optional[Callable]):
        return super()._create_validation_iterator(train, valid, n_folds, cv_iter)



