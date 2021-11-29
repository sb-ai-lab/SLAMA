import logging
from typing import Optional, Any

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from lightautoml.dataset.base import array_attr_roles, valid_array_attributes
from lightautoml.dataset.roles import ColumnRole, DropRole, NumericRole, DatetimeRole, CategoryRole
from lightautoml.dataset.utils import roles_parser
from lightautoml.reader.base import Reader, UserDefinedRolesDict, RoleType
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.tasks import Task

logger = logging.getLogger(__name__)


class SparkToSparkReader(Reader):
    """
    Reader to convert :class:`~pandas.DataFrame` to AutoML's :class:`~lightautoml.dataset.np_pd_dataset.PandasDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(
        self,
        task: Task,
        samples: Optional[int] = 100000,
        max_nan_rate: float = 0.999,
        max_constant_rate: float = 0.999,
        cv: int = 5,
        random_state: int = 42,
        roles_params: Optional[dict] = None,
        n_jobs: int = 4,
        # params for advanced roles guess
        advanced_roles: bool = True,
        numeric_unique_rate: float = 0.999,
        max_to_3rd_rate: float = 1.1,
        binning_enc_rate: float = 2,
        raw_decr_rate: float = 1.1,
        max_score_rate: float = 0.2,
        abs_score_val: float = 0.04,
        drop_score_co: float = 0.01,
        **kwargs: Any
    ):
        """

        Args:
            task: Task object.
            samples: Number of elements used when checking role type.
            max_nan_rate: Maximum nan-rate.
            max_constant_rate: Maximum constant rate.
            cv: CV Folds.
            random_state: Random seed.
            roles_params: dict of params of features roles. \
                Ex. {'numeric': {'dtype': np.float32}, 'datetime': {'date_format': '%Y-%m-%d'}}
                It's optional and commonly comes from config
            n_jobs: Int number of processes.
            advanced_roles: Param of roles guess (experimental, do not change).
            numeric_unqiue_rate: Param of roles guess (experimental, do not change).
            max_to_3rd_rate: Param of roles guess (experimental, do not change).
            binning_enc_rate: Param of roles guess (experimental, do not change).
            raw_decr_rate: Param of roles guess (experimental, do not change).
            max_score_rate: Param of roles guess (experimental, do not change).
            abs_score_val: Param of roles guess (experimental, do not change).
            drop_score_co: Param of roles guess (experimental, do not change).
            **kwargs: For now not used.

        """
        super().__init__(task)
        self.samples = samples
        self.max_nan_rate = max_nan_rate
        self.max_constant_rate = max_constant_rate
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.roles_params = roles_params
        self.target = None
        if roles_params is None:
            self.roles_params = {}

        self.advanced_roles = advanced_roles
        self.advanced_roles_params = {
            "numeric_unique_rate": numeric_unique_rate,
            "max_to_3rd_rate": max_to_3rd_rate,
            "binning_enc_rate": binning_enc_rate,
            "raw_decr_rate": raw_decr_rate,
            "max_score_rate": max_score_rate,
            "abs_score_val": abs_score_val,
            "drop_score_co": drop_score_co,
        }

        self.params = kwargs

    def fit_read(
        self, train_data: SparkDataFrame, features_names: Any = None, roles: UserDefinedRolesDict = None, **kwargs: Any
    ) -> SparkDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        logger.info("\x1b[1mTrain data shape: {}\x1b[0m\n".format(train_data.shape))

        if SparkDataset.ID_COLUMN not in train_data.columns:
            train_data = train_data.withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())
        train_data = train_data.cache()

        if roles is None:
            roles = {}

        # transform roles from user format {RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}
        # to automl format {'feat0': RoleX, 'feat1': RoleX, 'TARGET': RoleY, ...}
        parsed_roles = roles_parser(roles)
        # transform str role definition to automl ColumnRole
        attrs_dict = dict(zip(array_attr_roles, valid_array_attributes))

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if type(r) == str:
                # get default role params if defined
                r = self._get_default_role_from_str(r)

            # check if column is defined like target/group/weight etc ...
            if r.name in attrs_dict:
                # defined in kwargs is rewrited.. TODO: Maybe raise warning if rewrited?
                # TODO: Think, what if multilabel or multitask? Multiple column target ..
                # TODO: Maybe for multilabel/multitask make target only avaliable in kwargs??
                self._used_array_attrs[attrs_dict[r.name]] = feat
                kwargs[attrs_dict[r.name]] = train_data[feat]
                r = DropRole()

            # add new role
            parsed_roles[feat] = r

        assert "target" in kwargs, "Target should be defined"
        self.target = kwargs["target"]

        train_data = self._create_target(train_data, target_col=kwargs["target"])

        # get subsample if it needed
        subsample = train_data
        if self.samples:
            subsample = subsample.sample(fraction=0.1, seed=42).limit(self.samples).cache()

        # TODO: LAMA-SPARK rewrite this part:
        #   Implement the checking logic
        #   1. get spark df schema
        #   2. get feat in parsed roles
        #   3. parsed role matches Spark datatype ?
        #       if yes, just accept
        #       if no, try to test and convert
        #   4. there is no parsed role, try to match from Spark schema (datetime, ints, doubles)
        #   5. if not possible (e.g. string type), try to guess using LAMA logic

        # infer roles
        for feat in subsample.columns:
            if feat == SparkDataset.ID_COLUMN:
                continue

            assert isinstance(
                feat, str
            ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))
            if feat in parsed_roles:
                r = parsed_roles[feat]
                # handle datetimes
                if r.name == "Datetime":
                    # try if it's ok to infer date with given params
                    result = subsample.select(
                        F.to_timestamp(feat, format=r.format).isNotNull().astype(IntegerType()).alias(f"{feat}_dt"),
                        F.count('*')
                    ).first()

                    if result[f"{feat}_dt"] != result['count']:
                        raise ValueError("Looks like given datetime parsing params are not correctly defined")

                # replace default category dtype for numeric roles dtype if cat col dtype is numeric
                if r.name == "Category":
                    # default category role
                    cat_role = self._get_default_role_from_str("category")
                    # check if role with dtypes was exactly defined
                    flg_default_params = feat in roles["category"] if "category" in roles else False

                    inferred_dtype = next(dtyp for fname, dtyp in subsample.dtypes if fname == feat)
                    inferred_dtype = np.dtype(inferred_dtype)

                    if (
                        flg_default_params
                        and not np.issubdtype(cat_role.dtype, np.number)
                        and np.issubdtype(inferred_dtype, np.number)
                    ):
                        r.dtype = self._get_default_role_from_str("numeric").dtype

            else:
                # TODO: SPARK-LAMA this functions can be applied for all columns in a single round
                # if no - infer
                if self._is_ok_feature(subsample, feat):
                    r = self._guess_role(subsample, feat)
                else:
                    r = DropRole()

            # set back
            if r.name != "Drop":
                self._roles[feat] = r
                self._used_features.append(feat)
            else:
                self._dropped_features.append(feat)

        assert len(self.used_features) > 0, "All features are excluded for some reasons"
        # assert len(self.used_array_attrs) > 0, 'At least target should be defined in train dataset'
        # create folds

        # TODO: LAMA-SPARK remove it from here
        # we will use CrossValidation from Spark
        # folds = set_sklearn_folds(
        #     self.task,
        #     kwargs["target"].values,
        #     cv=self.cv,
        #     random_state=self.random_state,
        #     group=None if "group" not in kwargs else kwargs["group"],
        # )
        # if folds is not None:
        #     kwargs["folds"] = Series(folds, index=train_data.index)

        # get dataset
        # TODO: SPARK-LAMA send parent for unwinding
        dataset = SparkDataset(
            train_data.select(SparkDataset.ID_COLUMN, self.used_features),
            self.roles,
            task=self.task,
            **kwargs
        )

        # TODO: SPARK-LAMA will be implemented later
        # if self.advanced_roles:
        #     new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)
        #
        #     droplist = [x for x in new_roles if new_roles[x].name == "Drop" and not self._roles[x].force_input]
        #     self.upd_used_features(remove=droplist)
        #     self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
        #
        #     # TODO: SPARK-LAMA send parent for unwinding
        #     dataset = SparkDataset(
        #         train_data.select(SparkDataset.ID_COLUMN, self.used_features),
        #         self.roles,
        #         task=self.task,
        #         **kwargs
        #     )

        return dataset

    def _create_target(self, sdf: SparkDataFrame, target_col: str = "target"):
        """Validate target column and create class mapping is needed

        Args:
            target: Column with target values.

        Returns:
            Transformed target.

        """
        self.class_mapping = None

        rows = sdf.where(F.isnan(target_col)).first().collect()
        assert len(rows) == 0, "Nan in target detected"

        if self.task.name != "reg":
            srtd = sdf.select(target_col).distinct().sort().collect()
            srtd = np.array([r[target_col] for r in srtd])
            self._n_classes = len(srtd)

            if (np.arange(srtd.shape[0]) == srtd).all():

                assert srtd.shape[0] > 1, "Less than 2 unique values in target"
                if self.task.name == "binary":
                    assert srtd.shape[0] == 2, "Binary task and more than 2 values in target"
                return sdf

            self.class_mapping = {x: i for i, x in enumerate(srtd)}
            sdf_with_proc_target = sdf.na.replace(self.class_mapping, subset=[target_col])

            return sdf_with_proc_target

        return sdf

    def _get_default_role_from_str(self, name) -> RoleType:
        """Get default role for string name according to automl's defaults and user settings.

        Args:
            name: name of role to get.

        Returns:
            role object.

        """
        name = name.lower()
        try:
            role_params = self.roles_params[name]
        except KeyError:
            role_params = {}

        return ColumnRole.from_string(name, **role_params)

    def _guess_role(self, data: SparkDataFrame, feature: str) -> RoleType:
        """Try to infer role, simple way.

        If convertable to float -> number.
        Else if convertable to datetime -> datetime.
        Else category.

        Args:
            feature: Column from dataset.

        Returns:
            Feature role.

        """
        inferred_dtype = next(dtyp for fname, dtyp in data.dtypes if fname == feature)
        inferred_dtype = np.dtype(inferred_dtype)

        # testing if it can be numeric or not
        num_dtype = self._get_default_role_from_str("numeric").dtype
        date_format = self._get_default_role_from_str("datetime").format
        # TODO: can it be really converted?
        if np.issubdtype(inferred_dtype, np.number):
            return NumericRole(num_dtype)

        can_cast_to_numeric = F.col(feature).cast(num_dtype).isNotNull().astype(IntegerType())

        # TODO: utc handling here?
        can_cast_to_datetime = F.to_timestamp(feature, format=date_format).isNotNull().astype(IntegerType())

        result = data.select(
            F.sum(can_cast_to_datetime).alias(f"{feature}_num"),
            F.sum(can_cast_to_numeric).alias(f"{feature}_dt"),
            F.count('*').alias('count')
        ).first()

        if result[f"{feature}_num"] == result['count']:
            return NumericRole(num_dtype)

        if result[f"{feature}_dt"] == result['count']:
            return DatetimeRole(np.datetime64, date_format=date_format)

        return CategoryRole(object)

    def _is_ok_feature(self, train_data: SparkDataFrame, feature: str) -> bool:
        """Check if column is filled well to be a feature.

        Args:
            feature: Column from dataset.

        Returns:
            ``True`` if nan ratio and freqency are not high.

        """

        row = train_data.select(
            F.mean(F.isnan(feature).astype(IntegerType())).alias(f"{feature}_nan_rate"),
            (F.count_distinct(feature) / F.count(feature)).alias(f"{feature}_constant_rate")
        ).collect()[0]

        return (row[f"{feature}_nan_rate"] < self.max_nan_rate) \
               and (row[f"{feature}_constant_rate"] < self.max_constant_rate)

    def read(self, data: SparkDataFrame, features_names: Any = None, add_array_attrs: bool = False) -> SparkDataset:
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like
              target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """
        kwargs = {}
        target_col = "target"
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]

                if col_name not in data.columns:
                    continue

                if array_attr == "target" and self.class_mapping is not None:
                    data = data.na.replace(self.class_mapping, subset=[target_col])

                kwargs[array_attr] = target_col

        dataset = SparkDataset(
            data.select(SparkDataset.ID_COLUMN, target_col, *self.used_features),
            roles=self.roles,
            task=self.task,
            **kwargs
        )

        return dataset

    # TODO: SPARK-LAMA will be implemented later
    # def advanced_roles_guess(self, dataset: SparkDataset, manual_roles: Optional[RolesDict] = None) -> RolesDict:
    #     """Advanced roles guess over user's definition and reader's simple guessing.
    #
    #     Strategy - compute feature's NormalizedGini
    #     for different encoding ways and calc stats over results.
    #     Role is inferred by comparing performance stats with manual rules.
    #     Rule params are params of roles guess in init.
    #     Defaults are ok in general case.
    #
    #     Args:
    #         dataset: Input PandasDataset.
    #         manual_roles: Dict of user defined roles.
    #
    #     Returns:
    #         Dict.
    #
    #     """
    #     if manual_roles is None:
    #         manual_roles = {}
    #     top_scores = []
    #     new_roles_dict = dataset.roles
    #     advanced_roles_params = deepcopy(self.advanced_roles_params)
    #     drop_co = advanced_roles_params.pop("drop_score_co")
    #     # guess roles nor numerics
    #
    #     stat = get_numeric_roles_stat(
    #         dataset,
    #         manual_roles=manual_roles,
    #         random_state=self.random_state,
    #         subsample=self.samples,
    #         n_jobs=self.n_jobs,
    #     )
    #
    #     if len(stat) > 0:
    #         # upd stat with rules
    #
    #         stat = calc_encoding_rules(stat, **advanced_roles_params)
    #         new_roles_dict = {**new_roles_dict, **rule_based_roles_guess(stat)}
    #         top_scores.append(stat["max_score"])
    #     #
    #     # # # guess categories handling type
    #     stat = get_category_roles_stat(
    #         dataset,
    #         random_state=self.random_state,
    #         subsample=self.samples,
    #         n_jobs=self.n_jobs,
    #     )
    #     if len(stat) > 0:
    #         # upd stat with rules
    #         # TODO: add sample params
    #
    #         stat = calc_category_rules(stat)
    #         new_roles_dict = {**new_roles_dict, **rule_based_cat_handler_guess(stat)}
    #         top_scores.append(stat["max_score"])
    #     #
    #     # # get top scores of feature
    #     if len(top_scores) > 0:
    #         top_scores = pd.concat(top_scores, axis=0)
    #         # TODO: Add sample params
    #
    #         null_scores = get_null_scores(
    #             dataset,
    #             list(top_scores.index),
    #             random_state=self.random_state,
    #             subsample=self.samples,
    #         )
    #         top_scores = pd.concat([null_scores, top_scores], axis=1).max(axis=1)
    #         rejected = list(top_scores[top_scores < drop_co].index)
    #         logger.info3("Feats was rejected during automatic roles guess: {0}".format(rejected))
    #         new_roles_dict = {**new_roles_dict, **{x: DropRole() for x in rejected}}
    #
    #     return new_roles_dict