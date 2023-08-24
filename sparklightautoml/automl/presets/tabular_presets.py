import logging
import os

from copy import copy
from copy import deepcopy
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from lightautoml.automl.presets.base import upd_params
from lightautoml.automl.presets.utils import plot_pdp_with_distribution
from lightautoml.pipelines.selection.base import ComposedSelector
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.reader.tabular_batch_generator import ReadableToDf
from lightautoml.utils.timer import TaskTimer
from pyspark.ml import Transformer
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as sf
from pyspark.sql.types import DateType
from pyspark.sql.types import StringType
from tqdm import tqdm

from sparklightautoml.automl.base import ReadableIntoSparkDf
from sparklightautoml.automl.blend import SparkWeightedBlender
from sparklightautoml.automl.presets.base import SparkAutoMLPreset
from sparklightautoml.automl.presets.utils import calc_feats_permutation_imps
from sparklightautoml.automl.presets.utils import replace_dayofweek_in_date
from sparklightautoml.automl.presets.utils import replace_month_in_date
from sparklightautoml.automl.presets.utils import replace_year_in_date
from sparklightautoml.computations.builder import AutoMLComputationsSettings
from sparklightautoml.computations.builder import ComputationsManagerFactory
from sparklightautoml.dataset.base import PersistenceManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.ml_algo.tuning.parallel_optuna import ParallelOptunaTuner
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.pipelines.ml.nested_ml_pipe import SparkNestedTabularMLPipeline
from sparklightautoml.pipelines.selection.base import BugFixSelectionPipelineWrapper
from sparklightautoml.pipelines.selection.base import SparkSelectionPipelineWrapper
from sparklightautoml.pipelines.selection.permutation_importance_based import (
    SparkNpPermutationImportanceEstimator,
)
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import SparkDataFrame


logger = logging.getLogger(__name__)
base_dir = os.path.dirname(__file__)


class SparkTabularAutoML(SparkAutoMLPreset):
    """
    Spark version of :class:`TabularAutoML <lightautoml.automl.presets.tabular_presets.TabularAutoML>`.
    Represent high level entity of spark lightautoml. Use this class to create automl instance.

    Example:

        >>> automl = SparkTabularAutoML(
        >>>     spark=spark,
        >>>     task=SparkTask('binary'),
        >>>     general_params={"use_algos": [["lgb"]]},
        >>>     lgb_params={'use_single_dataset_mode': True},
        >>>     reader_params={"cv": cv, "advanced_roles": False}
        >>> )
        >>> oof_predictions = automl.fit_predict(
        >>>     train_data,
        >>>     roles=roles
        >>> )
    """

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
        task: SparkTask,
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
        persistence_manager: Optional[PersistenceManager] = None,
        computation_settings: AutoMLComputationsSettings = ("no_parallelism", -1),
    ):
        if config_path is None:
            config_path = os.path.join(base_dir, self._default_config_path)

        self._computation_managers_factory = (
            computation_settings
            if isinstance(computation_settings, ComputationsManagerFactory)
            else ComputationsManagerFactory(computation_settings)
        )

        super().__init__(
            task,
            timeout,
            memory_limit,
            cpu_limit,
            gpu_ids,
            timing_params,
            config_path,
            self._computation_managers_factory.get_ml_pipelines_manager(),
        )

        self._persistence_manager = persistence_manager or PlainCachePersistenceManager()

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

    def get_selector(self, n_level: Optional[int] = 1) -> SparkSelectionPipelineWrapper:
        selection_params = self.selection_params
        # lgb_params
        lgb_params = deepcopy(self.lgb_params)
        lgb_params["default_params"] = {
            **lgb_params["default_params"],
            **{"featureFraction": 1},
        }

        mode = selection_params["mode"]

        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            # timer will be useful to estimate time for next gbm runs
            time_score = self.get_time_score(n_level, "lgb", False)

            sel_timer_0 = self.timer.get_task_timer("lgb", time_score)
            selection_feats = SparkLGBSimpleFeatures()

            selection_gbm = SparkBoostLGBM(timer=sel_timer_0, **lgb_params)
            selection_gbm.set_prefix("Selector")

            if selection_params["importance_type"] == "permutation":
                importance = SparkNpPermutationImportanceEstimator()
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
                selection_feats = SparkLGBSimpleFeatures()
                selection_gbm = SparkBoostLGBM(timer=sel_timer_1, **lgb_params)
                selection_gbm.set_prefix("Selector")

                importance = SparkNpPermutationImportanceEstimator(
                    computations_settings=self._computation_managers_factory.get_selector_manager()
                )

                extra_selector = NpIterativeFeatureSelector(
                    selection_feats,
                    selection_gbm,
                    importance,
                    feature_group_size=selection_params["feature_group_size"],
                    max_features_cnt_in_result=selection_params["max_features_cnt_in_result"],
                )

                pre_selector = ComposedSelector([pre_selector, extra_selector])

        pre_selector = BugFixSelectionPipelineWrapper(pre_selector)
        return SparkSelectionPipelineWrapper(pre_selector)

    def get_linear(
        self, n_level: int = 1, pre_selector: Optional[SparkSelectionPipelineWrapper] = None
    ) -> SparkNestedTabularMLPipeline:
        # linear model with l2
        time_score = self.get_time_score(n_level, "linear_l2")
        linear_l2_timer = self.timer.get_task_timer("reg_l2", time_score)
        linear_l2_params = {**self.linear_l2_params}

        linear_l2_model = SparkLinearLBFGS(
            timer=linear_l2_timer,
            computations_settings=self._computation_managers_factory.get_linear_manager(),
            **linear_l2_params,
        )
        linear_l2_feats = SparkLinearFeatures(output_categories=True, **self.linear_pipeline_params)

        linear_l2_pipe = SparkNestedTabularMLPipeline(
            [linear_l2_model],
            force_calc=True,
            pre_selection=pre_selector,
            features_pipeline=linear_l2_feats,
            computations_settings=self._computation_managers_factory.get_ml_algo_manager(),
            **self.nested_cv_params,
        )
        return linear_l2_pipe

    def get_gbms(
        self,
        keys: Sequence[str],
        n_level: int = 1,
        pre_selector: Optional[SparkSelectionPipelineWrapper] = None,
    ):
        gbm_feats = SparkLGBAdvancedPipeline(**self.gbm_pipeline_params)

        ml_algos = []
        force_calc = []
        for key, force in zip(keys, [True, False, False, False]):
            tuned = "_tuned" in key
            algo_key = key.split("_")[0]
            time_score = self.get_time_score(n_level, key)
            gbm_timer = self.timer.get_task_timer(algo_key, time_score)
            gbm_model, lgb_params = self._get_boosting_model(algo_key, gbm_timer)

            if tuned:
                gbm_model.set_prefix("Tuned")

                tuner_comp_manager = self._computation_managers_factory.get_tuning_manager()

                gbm_tuner = ParallelOptunaTuner(
                    n_trials=self.tuning_params["max_tuning_iter"],
                    timeout=self.tuning_params["max_tuning_time"],
                    fit_on_holdout=self.tuning_params["fit_on_holdout"],
                    parallelism=tuner_comp_manager.parallelism,
                    computations_manager=tuner_comp_manager,
                )
                gbm_model = (gbm_model, gbm_tuner)

            ml_algos.append(gbm_model)
            force_calc.append(force)

        gbm_pipe = SparkNestedTabularMLPipeline(
            ml_algos,
            force_calc,
            pre_selection=pre_selector,
            features_pipeline=gbm_feats,
            computations_settings=self._computation_managers_factory.get_ml_algo_manager(),
            **self.nested_cv_params,
        )

        return gbm_pipe

    def _get_boosting_model(self, algo_key: str, gbm_timer: Optional[TaskTimer] = None):
        if algo_key == "lgb":
            lgb_params = {**self.lgb_params}

            gbm_model = SparkBoostLGBM(
                timer=gbm_timer,
                computations_settings=self._computation_managers_factory.get_lgb_manager(),
                **lgb_params,
            )
            return gbm_model, lgb_params
        elif algo_key == "cb":
            raise NotImplementedError("Not supported yet")
        else:
            raise ValueError("Wrong algo key")

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
                x for x in ["lgb", "lgb_tuned", "cb", "cb_tuned"] if x in names and x.split("_")[0] in self.task.losses
            ]

            if len(gbm_models) > 0:
                selector = None
                if "gbm" in self.selection_params["select_algos"] and (self.general_params["skip_conn"] or n == 0):
                    selector = pre_selector
                lvl.append(self.get_gbms(gbm_models, n + 1, selector))

            levels.append(lvl)

        # blend everything
        blender = SparkWeightedBlender(max_nonzero_coef=self.general_params["weighted_blender_max_nonzero_coef"])

        # initialize
        self._initialize(
            reader,
            levels,
            skip_conn=self.general_params["skip_conn"],
            blender=blender,
            return_all_predictions=self.general_params["return_all_predictions"],
            timer=self.timer,
        )

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
        persistence_manager: Optional[PersistenceManager] = None,
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
        train = self._read_data(train_data, train_features)
        # if upd_roles:
        #     roles = {**roles, **upd_roles}
        if valid_data is not None:
            valid_data = self._read_data(valid_data, valid_features)

        oof_pred = super().fit_predict(
            train,
            roles=roles,
            cv_iter=cv_iter,
            valid_data=valid_data,
            verbose=verbose,
            persistence_manager=persistence_manager,
        )

        return oof_pred

    def get_feature_scores(
        self,
        calc_method: str = "fast",
        data: Optional[ReadableIntoSparkDf] = None,
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
        data = self._read_data(data, features_names, read_csv_params)
        used_feats = self.collect_used_feats()
        fi = calc_feats_permutation_imps(
            self,
            used_feats,
            data,
            self.task.get_dataset_metric(),
            silent=silent,
        )
        return fi

    @staticmethod
    def get_histogram(data: SparkDataFrame, column: str, n_bins: int) -> Tuple[List, np.ndarray]:
        assert n_bins >= 2, "n_bins must be equal 2 or more"
        bin_edges, counts = (
            data.select(sf.col(column).cast("double"))
            .where(sf.col(column).isNotNull())
            .rdd.map(lambda x: x[0])
            .histogram(n_bins)
        )
        bin_edges = np.array(bin_edges)
        return counts, bin_edges

    @staticmethod
    def get_pdp_data_numeric_feature(
        df: SparkDataFrame,
        feature_name: str,
        model: Transformer,
        prediction_col: str,
        n_bins: int,
        ice_fraction: float = 1.0,
        ice_fraction_seed: int = 42,
    ) -> Tuple[List, List, List]:
        """Returns `grid`, `ys` and `counts` calculated on input numeric column to plot PDP.

        Args:
            df (SparkDataFrame): Spark DataFrame with `feature_name` column
            feature_name (str): feature column name
            model (PipelineModel): Spark Pipeline Model
            prediction_col (str): prediction column to be created by the `model`
            n_bins (int): The number of bins to produce. Raises exception if n_bins < 2.
            ice_fraction (float, optional): What fraction of the input dataframe will be used to make predictions.
                Useful for very large dataframe. Defaults to 1.0.
            ice_fraction_seed (int, optional): Seed for `ice_fraction`. Defaults to 42.

        Returns:
            Tuple[List, List, List]:
            `grid` is list of categories,
            `ys` is list of predictions by category,
            `counts` is numbers of values by category
        """
        counts, bin_edges = SparkTabularAutoML.get_histogram(df, feature_name, n_bins)
        grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        ys = []
        sample_df = (
            df.select(*[c for c in df.columns if c != feature_name])
            .sample(fraction=ice_fraction, seed=ice_fraction_seed)
            .cache()
        )
        for i in tqdm(grid):
            # replace feature column values with constant
            sdf = sample_df.select("*", sf.lit(i).alias(feature_name))

            # infer via transformer
            preds = model.transform(sdf)
            # TODO: SPARK-LAMA remove this line after passing the "prediction_col" parameter
            prediction_col = next(c for c in preds.columns if c.startswith("prediction"))
            preds = np.array(preds.select(prediction_col).collect())
            # when preds.shape is (n, 1, k) we change it to (n, k),
            # where n is number of rows and k is number of classes
            if len(preds.shape) == 3:
                preds = np.squeeze(preds, axis=1)
            ys.append(preds)

        sample_df.unpersist()

        return grid, ys, counts

    @staticmethod
    def get_pdp_data_categorical_feature(
        df: SparkDataFrame,
        feature_name: str,
        model: Transformer,
        prediction_col: str,
        n_top_cats: int,
        ice_fraction: float = 1.0,
        ice_fraction_seed: int = 42,
    ) -> Tuple[List, List, List]:
        """Returns `grid`, `ys` and `counts` calculated on input categorical column to plot PDP.

        Args:
            df (SparkDataFrame): Spark DataFrame with `feature_name` column
            feature_name (str): feature column name
            model (PipelineModel): Spark Pipeline Model
            prediction_col (str): prediction column to be created by the `model`
            n_top_cats (int): param to selection top n categories
            ice_fraction (float, optional): What fraction of the input dataframe will be used to make predictions.
                Useful for very large dataframe. Defaults to 1.0.
            ice_fraction_seed (int, optional): Seed for `ice_fraction`. Defaults to 42.

        Returns:
            Tuple[List, List, List]:
            `grid` is list of categories,
            `ys` is list of predictions by category,
            `counts` is numbers of values by category
        """
        feature_cnt = (
            df.where(sf.col(feature_name).isNotNull()).groupBy(feature_name).count().orderBy(sf.desc("count")).collect()
        )
        grid = [row[feature_name] for row in feature_cnt[:n_top_cats]]
        counts = [row["count"] for row in feature_cnt[:n_top_cats]]
        ys = []
        sample_df = (
            df.select(*[c for c in df.columns if c != feature_name])
            .sample(fraction=ice_fraction, seed=ice_fraction_seed)
            .cache()
        )
        for i in tqdm(grid):
            sdf = sample_df.select("*", sf.lit(i).alias(feature_name))
            preds = model.transform(sdf)
            # TODO: SPARK-LAMA remove this line after passing the "prediction_col" parameter
            prediction_col = next(c for c in preds.columns if c.startswith("prediction"))
            preds = np.array(preds.select(prediction_col).collect())
            # when preds.shape is (n, 1, k) we change it to (n, k),
            # where n is number of rows and k is number of classes
            if len(preds.shape) == 3:
                preds = np.squeeze(preds, axis=1)
            ys.append(preds)
        if len(feature_cnt) > n_top_cats:
            # unique other categories
            unique_other_categories = [row[feature_name] for row in feature_cnt[n_top_cats:]]

            # get non-top categories, natural distributions is important here
            w = Window().orderBy(sf.lit("A"))  # window without sorting
            other_categories_collection = (
                df.select(feature_name)
                .filter(sf.col(feature_name).isin(unique_other_categories))
                .select(sf.row_number().over(w).alias("row_num"), feature_name)
                .collect()
            )

            # dict with key=%row number% and value=%category%
            other_categories_dict = {x["row_num"]: x[feature_name] for x in other_categories_collection}
            max_row_num = len(other_categories_collection)

            def get_category_by_row_num(row_num):
                remainder = row_num % max_row_num
                if remainder == 0:
                    key = max_row_num
                else:
                    key = remainder
                return other_categories_dict[key]

            get_category_udf = sf.udf(get_category_by_row_num, StringType())

            # add row number to main dataframe and exclude feature_name column
            sdf = sample_df.select("*", sf.row_number().over(w).alias("row_num"))

            all_columns_except_row_num = [f for f in sdf.columns if f != "row_num"]
            feature_col = get_category_udf(sf.col("row_num")).alias(feature_name)
            # exclude row number from dataframe
            # and add back feature_name column filled with other categories same distribution
            sdf = sdf.select(*all_columns_except_row_num, feature_col)

            preds = model.transform(sdf)
            preds = np.array(preds.select(prediction_col).collect())
            # when preds.shape is (n, 1, k) we change it to (n, k),
            # where n is number of rows and k is number of classes
            if len(preds.shape) == 3:
                preds = np.squeeze(preds, axis=1)

            grid.append("<OTHER>")
            ys.append(preds)
            counts.append(sum([row["count"] for row in feature_cnt[n_top_cats:]]))

        sample_df.unpersist()

        return grid, ys, counts

    @staticmethod
    def get_pdp_data_datetime_feature(
        df: SparkDataFrame,
        feature_name: str,
        model: Transformer,
        prediction_col: str,
        datetime_level: str,
        reader,
        ice_fraction: float = 1.0,
        ice_fraction_seed: int = 42,
    ) -> Tuple[List, List, List]:
        """Returns `grid`, `ys` and `counts` calculated on input datetime column to plot PDP.

        Args:
            df (SparkDataFrame): Spark DataFrame with `feature_name` column
            feature_name (str): feature column name
            model (PipelineModel): Spark Pipeline Model
            prediction_col (str): prediction column to be created by the `model`
            datetime_level (str): Unit of time that will be modified to calculate dependence: "year", "month" or "dayofweek"
            reader (_type_): Automl reader to transform input dataframe before `model` inferring.
            ice_fraction (float, optional): What fraction of the input dataframe will be used to make predictions.
                Useful for very large dataframe. Defaults to 1.0.
            ice_fraction_seed (int, optional): Seed for `ice_fraction`. Defaults to 42.

        Returns:
            Tuple[List, List, List]:
            `grid` is list of categories,
            `ys` is list of predictions by category,
            `counts` is numbers of values by category
        """
        df = reader.read(df).data
        if datetime_level == "year":
            feature_cnt = df.groupBy(sf.year(feature_name).alias("year")).count().orderBy(sf.asc("year")).collect()
            grid = [x["year"] for x in feature_cnt]
            counts = [row["count"] for row in feature_cnt]
            replace_date_element_udf = sf.udf(replace_year_in_date, DateType())
        elif datetime_level == "month":
            feature_cnt = df.groupBy(sf.month(feature_name).alias("month")).count().orderBy(sf.asc("month")).collect()
            grid = np.arange(1, 13)
            grid = grid.tolist()
            counts = [0] * 12
            for row in feature_cnt:
                counts[row["month"] - 1] = row["count"]
            replace_date_element_udf = sf.udf(replace_month_in_date, DateType())
        else:
            feature_cnt = (
                df.groupBy(sf.dayofweek(feature_name).alias("dayofweek")).count().orderBy(sf.asc("dayofweek")).collect()
            )
            grid = np.arange(7)
            grid = grid.tolist()
            counts = [0] * 7
            for row in feature_cnt:
                counts[row["dayofweek"] - 1] = row["count"]
            replace_date_element_udf = sf.udf(replace_dayofweek_in_date, DateType())
        ys = []
        sample_df = df.sample(fraction=ice_fraction, seed=ice_fraction_seed).cache()
        for i in tqdm(grid):
            feature_col = replace_date_element_udf(sf.col(feature_name), sf.lit(i)).alias(feature_name)
            sdf = sample_df.select(*[c for c in sample_df.columns if c != feature_name], feature_col)
            preds = model.transform(sdf)
            # TODO: SPARK-LAMA remove this line after passing the "prediction_col" parameter
            prediction_col = next(c for c in preds.columns if c.startswith("prediction"))
            preds = np.array(preds.select(prediction_col).collect())
            # when preds.shape is (n, 1, k) we change it to (n, k),
            # where n is number of rows and k is number of classes
            if len(preds.shape) == 3:
                preds = np.squeeze(preds, axis=1)
            ys.append(preds)

        return grid, ys, counts

    def get_individual_pdp(
        self,
        test_data: SparkDataFrame,
        feature_name: str,
        n_bins: Optional[int] = 30,
        top_n_categories: Optional[int] = 10,
        datetime_level: Optional[str] = "year",
        ice_fraction: float = 1.0,
        ice_fraction_seed: int = 42,
    ):
        assert feature_name in self.reader._roles
        assert datetime_level in ["year", "month", "dayofweek"]
        assert 0 < ice_fraction <= 1.0

        pipeline_model = self.transformer(leave_only_predict_cols=True)

        # Numerical features
        if self.reader._roles[feature_name].name == "Numeric":
            return self.get_pdp_data_numeric_feature(
                test_data, feature_name, pipeline_model, "prediction", n_bins, ice_fraction, ice_fraction_seed
            )
        # Categorical features
        elif self.reader._roles[feature_name].name == "Category":
            return self.get_pdp_data_categorical_feature(
                test_data, feature_name, pipeline_model, "prediction", top_n_categories, ice_fraction, ice_fraction_seed
            )
        # Datetime Features
        elif self.reader._roles[feature_name].name == "Datetime":
            return self.get_pdp_data_datetime_feature(
                test_data,
                feature_name,
                pipeline_model,
                "prediction",
                datetime_level,
                self.reader,
                ice_fraction,
                ice_fraction_seed,
            )
        else:
            raise ValueError("Supported only Numeric, Category or Datetime feature")

    def plot_pdp(
        self,
        test_data: ReadableToDf,
        feature_name: str,
        individual: Optional[bool] = False,
        n_bins: Optional[int] = 30,
        top_n_categories: Optional[int] = 10,
        top_n_classes: Optional[int] = 10,
        datetime_level: Optional[str] = "year",
        ice_fraction: float = 1.0,
        ice_fraction_seed: int = 42,
    ):
        grid, ys, counts = self.get_individual_pdp(
            test_data=test_data,
            feature_name=feature_name,
            n_bins=n_bins,
            top_n_categories=top_n_categories,
            datetime_level=datetime_level,
            ice_fraction=ice_fraction,
            ice_fraction_seed=ice_fraction_seed,
        )

        histogram_data_rows_limit = 2000
        rows_count = test_data.count()
        if rows_count > histogram_data_rows_limit:
            fraction = histogram_data_rows_limit / rows_count
            test_data = test_data.sample(frac=fraction)
        if self.reader._roles[feature_name].name == "Numeric":
            test_data = test_data.select(sf.col(feature_name).cast("double")).toPandas()
        else:
            test_data = test_data.select(feature_name).toPandas()
        plot_pdp_with_distribution(
            test_data,
            grid,
            ys,
            counts,
            self.reader,
            feature_name,
            individual,
            top_n_classes,
            datetime_level,
        )
