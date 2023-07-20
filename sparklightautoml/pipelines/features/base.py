"""Basic classes for features generation."""
import itertools
import logging
import uuid

from copy import copy
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import cast

import numpy as np
import toposort

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.utils import get_columns_by_role
from pandas import DataFrame
from pandas import Series
from pyspark.ml import Estimator
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer
from pyspark.sql import functions as sf

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.mlwriters import CommonPickleMLReadable
from sparklightautoml.mlwriters import CommonPickleMLWritable
from sparklightautoml.pipelines.base import TransformerInputOutputRoles
from sparklightautoml.transformers.base import SparkBaseEstimator
from sparklightautoml.transformers.base import SparkBaseTransformer
from sparklightautoml.transformers.base import SparkChangeRolesTransformer
from sparklightautoml.transformers.base import SparkColumnsAndRoles
from sparklightautoml.transformers.base import SparkEstOrTrans
from sparklightautoml.transformers.base import SparkSequentialTransformer
from sparklightautoml.transformers.base import SparkUnionTransformer
from sparklightautoml.transformers.categorical import SparkCatIntersectionsEstimator
from sparklightautoml.transformers.categorical import SparkFreqEncoderEstimator
from sparklightautoml.transformers.categorical import SparkLabelEncoderEstimator
from sparklightautoml.transformers.categorical import (
    SparkMulticlassTargetEncoderEstimator,
)
from sparklightautoml.transformers.categorical import SparkOrdinalEncoderEstimator
from sparklightautoml.transformers.categorical import SparkTargetEncoderEstimator
from sparklightautoml.transformers.datetime import SparkBaseDiffTransformer
from sparklightautoml.transformers.datetime import SparkDateSeasonsEstimator
from sparklightautoml.transformers.numeric import SparkQuantileBinningEstimator
from sparklightautoml.utils import Cacher
from sparklightautoml.utils import ColumnsSelectorTransformer
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.utils import warn_if_not_cached


logger = logging.getLogger(__name__)


def build_graph(begin: SparkEstOrTrans):
    """Fill dict that represents graph of estimators and transformers

    Args:
        begin (SparkEstOrTrans): pipeline to extract graph of estimators and transformers
    """
    graph = dict()

    def find_start_end(tr: SparkEstOrTrans) -> Tuple[List[SparkEstOrTrans], List[SparkEstOrTrans]]:
        if isinstance(tr, SparkSequentialTransformer):
            se = [st_or_end for el in tr.transformers for st_or_end in find_start_end(el)]

            starts = se[0]
            ends = se[-1]
            middle = se[1:-1]

            i = 0
            while i < len(middle):
                for new_st, new_end in itertools.product(middle[i], middle[i + 1]):
                    if new_end not in graph:
                        graph[new_end] = set()
                    graph[new_end].add(new_st)
                i += 2

            return starts, ends

        elif isinstance(tr, SparkUnionTransformer):
            se = [find_start_end(el) for el in tr.transformers]
            starts = [s_el for s, _ in se for s_el in s]
            ends = [e_el for _, e in se for e_el in e]
            return starts, ends
        else:
            return [tr], [tr]

    init_starts, final_ends = find_start_end(begin)

    for st in init_starts:
        if st not in graph:
            graph[st] = set()

    return graph


@dataclass
class FittedPipe:
    dataset: SparkDataset
    transformer: Transformer


class SparkFeaturesPipeline(FeaturesPipeline, TransformerInputOutputRoles):
    """Abstract class.

    Analyze train dataset and create composite transformer
    based on subset of features.
    Instance can be interpreted like Transformer
    (look for :class:`~lightautoml.transformers.base.LAMLTransformer`)
    with delayed initialization (based on dataset metadata)
    Main method, user should define in custom pipeline is ``.create_pipeline``.
    For example, look at
    :class:`~lightautoml.pipelines.features.lgb_pipeline.LGBSimpleFeatures`.
    After FeaturePipeline instance is created, it is used like transformer
    with ``.fit_transform`` and ``.transform`` method.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipes: List[Callable[[SparkDataset], SparkEstOrTrans]] = [self.create_pipeline]
        self._transformer: Optional[Transformer] = None
        self._input_roles: Optional[RolesDict] = None
        self._output_roles: Optional[RolesDict] = None
        self._service_columns: Optional[List[str]] = None

    @property
    def input_features(self) -> List[str]:
        return list(self.input_roles.keys())

    @input_features.setter
    def input_features(self, val: List[str]):
        """Setter for input_features.

        Args:
            val: List of strings.

        """
        raise NotImplementedError("Unsupported operation")

    @property
    def output_features(self) -> List[str]:
        return list(self.output_roles.keys())

    @property
    def input_roles(self) -> Optional[RolesDict]:
        return self._input_roles

    @property
    def output_roles(self) -> Optional[RolesDict]:
        return self._output_roles

    def _get_service_columns(self) -> List[str]:
        return self._service_columns

    def _build_transformer(self, *args, **kwargs) -> Optional[Transformer]:
        return self._transformer

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
        """Analyse dataset and create composite transformer.

        Args:
            train: Dataset with train data.

        Returns:
            Composite transformer (pipeline).

        """
        raise NotImplementedError

    def fit_transform(self, train: SparkDataset) -> SparkDataset:
        """Create pipeline and then fit on train data and then transform.

        Args:
            train: Dataset with train data.n

        Returns:
            Dataset with new features.

        """
        logger.info("SparkFeaturePipeline is started")

        fitted_pipe = self._merge_pipes(train)
        self._transformer = fitted_pipe.transformer
        self._input_roles = copy(train.roles)
        self._output_roles = copy(fitted_pipe.dataset.roles)
        self._service_columns = train.service_columns

        logger.info("SparkFeaturePipeline is finished")

        return fitted_pipe.dataset

    def transform(self, test: SparkDataset) -> SparkDataset:
        return self._make_transformed_dataset(test)

    def append(self, pipeline):
        if isinstance(pipeline, SparkFeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in pipeline:
            self.pipes.extend(_pipeline.pipes)

        return self

    def prepend(self, pipeline):
        if isinstance(pipeline, SparkFeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in reversed(pipeline):
            self.pipes = _pipeline.pipes + self.pipes

        return self

    def pop(self, i: int = -1) -> Optional[Callable[[SparkDataset], Estimator]]:
        if len(self.pipes) > 1:
            return self.pipes.pop(i)

    def _merge_pipes(self, data: SparkDataset) -> FittedPipe:
        fitted_pipes = [self._optimize_and_fit(data, pipe(data)) for pipe in self.pipes]

        processed_dataset = SparkDataset.concatenate([fp.dataset for fp in fitted_pipes], name=f"{type(self)}")
        pipeline = PipelineModel(stages=[fp.transformer for fp in fitted_pipes])

        return FittedPipe(dataset=processed_dataset, transformer=pipeline)

    def _optimize_and_fit(self, train: SparkDataset, pipeline: SparkEstOrTrans) -> FittedPipe:
        graph = build_graph(pipeline)
        tr_layers = list(toposort.toposort(graph))

        cacher_key = f"{type(self)}_{uuid.uuid4()}"

        logger.info(f"Number of layers in the current feature pipeline {self}: {len(tr_layers)}")

        def exit_nodes(gr: Dict) -> Set:
            parents = set(el for v in gr.values() for el in v)
            all_nodes = set(gr.keys())
            found_exit_nodes = all_nodes.difference(parents)
            return found_exit_nodes

        def out_cols(est: SparkColumnsAndRoles) -> List[str]:
            cols = est.getOutputCols()
            if len(cols) > 0:
                return cols
            return [f"{est.get_prefix()}*"]

        def cum_outputs_layers(external_input: Set[str], layers):
            available_inputs = [external_input]
            for layer in layers:
                outs = {col for est in layer for col in out_cols(est)}
                available_inputs.append(available_inputs[-1].union(outs))

            return available_inputs

        def cum_inputs_layers(layers):
            layers = list(reversed(layers))
            available_inputs = [{col for est in layers[0] for col in out_cols(est)}]
            for layer in layers[1:]:
                outs = {col for est in layer for col in est.getInputCols()}
                available_inputs.append(available_inputs[-1].union(outs))
            return list(reversed(available_inputs))

        enodes = exit_nodes(graph)
        out_deps = cum_outputs_layers(set(train.features), tr_layers)
        in_deps = cum_inputs_layers([*tr_layers, enodes])
        cols_to_select_in_layers = [
            list(out_feats.intersection(next_in_feats)) for out_feats, next_in_feats in zip(out_deps[1:], in_deps[1:])
        ]

        dag_pipeline = Pipeline(
            stages=[
                stage
                for i, (layer, cols) in enumerate(zip(tr_layers, cols_to_select_in_layers))
                for stage in itertools.chain(
                    layer,
                    [
                        ColumnsSelectorTransformer(
                            name=f"{type(self).__name__} Layer: {i}",
                            input_cols=[SparkDataset.ID_COLUMN, *cols],
                            optional_cols=[c for c in train.service_columns if c != SparkDataset.ID_COLUMN],
                            transform_only_first_time=True,
                        ),
                        Cacher(cacher_key),
                    ],
                )
            ]
        )

        dag_transformer = dag_pipeline.fit(train.data)

        feature_sdf = Cacher.get_dataset_by_key(cacher_key)
        output_roles = {feat: role for est in enodes for feat, role in est.get_output_roles().items()}
        featurized_train = train.empty()
        featurized_train.set_data(
            feature_sdf,
            list(output_roles.keys()),
            output_roles,
            dependencies=[lambda: Cacher.release_cache_by_key(cacher_key), train],
            name=type(self).__name__,
        )

        return FittedPipe(dataset=featurized_train, transformer=dag_transformer)


class SparkTabularDataFeatures:
    """Helper class contains basic features transformations for tabular data.

    This method can de shared by all tabular feature pipelines,
    to simplify ``.create_automl`` definition.
    """

    def __init__(self, **kwargs: Any):
        """Set default parameters for tabular pipeline constructor.

        Args:
            **kwargs: Additional parameters.

        """
        self.multiclass_te_co = 3
        self.top_intersections = 5
        self.max_intersection_depth = 3
        self.subsample = 0.1  # 10000
        self.random_state = 42
        self.feats_imp = None
        self.ascending_by_cardinality = False

        self.max_bin_count = 10
        self.sparse_ohe = "auto"

        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    # noinspection PyMethodMayBeStatic
    def _cols_by_role(self, dataset: SparkDataset, role_name: str, **kwargs: Any) -> List[str]:
        cols = get_columns_by_role(dataset, role_name, **kwargs)
        filtered_cols = [col for col in cols]
        return filtered_cols

    def get_cols_for_datetime(self, train: SparkDataset) -> Tuple[List[str], List[str]]:
        """Get datetime columns to calculate features.

        Args:
            train: Dataset with train data.

        Returns:
            2 list of features names - base dates and common dates.

        """
        base_dates = self._cols_by_role(train, "Datetime", base_date=True)
        datetimes = self._cols_by_role(train, "Datetime", base_date=False) + self._cols_by_role(
            train, "Datetime", base_date=True, base_feats=True
        )

        return base_dates, datetimes

    def get_datetime_diffs(self, train: SparkDataset) -> Optional[SparkBaseTransformer]:
        """Difference for all datetimes with base date.

        Args:
            train: Dataset with train data.

        Returns:
            Transformer or ``None`` if no required features.

        """
        base_dates, datetimes = self.get_cols_for_datetime(train)
        if len(datetimes) == 0 or len(base_dates) == 0:
            return None

        roles = {f: train.roles[f] for f in itertools.chain(base_dates, datetimes)}

        base_diff = SparkBaseDiffTransformer(input_roles=roles, base_names=base_dates, diff_names=datetimes)

        return base_diff

    def get_datetime_seasons(
        self, train: SparkDataset, outp_role: Optional[ColumnRole] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get season params from dates.

        Args:
            train: Dataset with train data.
            outp_role: Role associated with output features.

        Returns:
            Transformer or ``None`` if no required features.

        """
        _, datetimes = self.get_cols_for_datetime(train)
        for col in copy(datetimes):
            role = cast(DatetimeRole, train.roles[col])
            if len(role.seasonality) == 0 and role.country is None:
                datetimes.remove(col)

        if len(datetimes) == 0:
            return

        if outp_role is None:
            outp_role = NumericRole(np.float32)

        roles = {f: train.roles[f] for f in datetimes}

        date_as_cat = SparkDateSeasonsEstimator(input_cols=datetimes, input_roles=roles, output_role=outp_role)

        return date_as_cat

    def get_numeric_data(
        self,
        train: SparkDataset,
        feats_to_select: Optional[List[str]] = None,
        prob: Optional[bool] = None,
    ) -> Optional[SparkBaseTransformer]:
        """Select numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.
            prob: Probability flag.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            if prob is None:
                feats_to_select = self._cols_by_role(train, "Numeric")
            else:
                feats_to_select = self._cols_by_role(train, "Numeric", prob=prob)

        if len(feats_to_select) == 0:
            return None

        roles = {f: train.roles[f] for f in feats_to_select}

        num_processing = SparkChangeRolesTransformer(
            input_cols=feats_to_select, input_roles=roles, role=NumericRole(np.float32)
        )

        return num_processing

    def get_numeric_vectors_data(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None, prob: Optional[bool] = None
    ):
        if feats_to_select is None:
            if prob is None:
                feats_to_select = self._cols_by_role(train, "NumericVectorOrArray")
            else:
                feats_to_select = self._cols_by_role(train, "NumericVectorOrArray", prob=prob)

        if len(feats_to_select) == 0:
            return None

        roles = cast(Dict[str, NumericVectorOrArrayRole], {f: train.roles[f] for f in feats_to_select})

        new_roles = {
            feat: NumericVectorOrArrayRole(
                role.size, role.element_col_name_template, np.float32, is_vector=role.is_vector
            )
            for feat, role in roles.items()
        }

        num_processing = SparkChangeRolesTransformer(input_cols=feats_to_select, input_roles=roles, roles=new_roles)

        return num_processing

    def get_freq_encoding(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get frequency encoding part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Category", encoding_type="freq")

        if len(feats_to_select) == 0:
            return None

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkFreqEncoderEstimator(input_cols=feats_to_select, input_roles=roles)

        return cat_processing

    def get_ordinal_encoding(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get order encoded part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Category", ordinal=True)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        ord_est = SparkOrdinalEncoderEstimator(
            input_cols=feats_to_select, input_roles=roles, subs=self.subsample, random_state=self.random_state
        )

        return ord_est

    def get_categorical_raw(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get label encoded categories data.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """

        if feats_to_select is None:
            feats_to_select = []
            for i in ["auto", "oof", "int", "ohe"]:
                feats = self._cols_by_role(train, "Category", encoding_type=i)
                feats_to_select.extend(feats)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkLabelEncoderEstimator(
            input_cols=feats_to_select, input_roles=roles, subs=self.subsample, random_state=self.random_state
        )
        return cat_processing

    def get_target_encoder(self, train: SparkDataset) -> Optional[type]:
        """Get target encoder func for dataset.

        Args:
            train: Dataset with train data.

        Returns:
            Class

        """
        target_encoder = None
        if train.folds_column is not None:
            if train.task.name in ["binary", "reg"]:
                target_encoder = SparkTargetEncoderEstimator
            else:
                result = train.data.select(sf.max(train.target_column).alias("max")).first()
                n_classes = result["max"] + 1

                if n_classes <= self.multiclass_te_co:
                    target_encoder = SparkMulticlassTargetEncoderEstimator

        return target_encoder

    def get_binned_data(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get encoded quantiles of numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: features to hanlde. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Numeric", discretization=True)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        binned_processing = SparkQuantileBinningEstimator(
            input_cols=feats_to_select, input_roles=roles, nbins=self.max_bin_count
        )

        return binned_processing

    def get_categorical_intersections(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get transformer that implements categorical intersections.

        Args:
            train: Dataset with train data.
            feats_to_select: features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """

        if feats_to_select is None:
            categories = get_columns_by_role(train, "Category")
            feats_to_select = categories

            if len(categories) <= 1:
                return

            elif len(categories) > self.top_intersections:
                feats_to_select = self.get_top_categories(train, self.top_intersections)

        elif len(feats_to_select) <= 1:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkCatIntersectionsEstimator(
            input_cols=feats_to_select, input_roles=roles, max_depth=self.max_intersection_depth
        )

        return cat_processing

    # noinspection PyMethodMayBeStatic
    def get_uniques_cnt(self, train: SparkDataset, feats: List[str]) -> Series:
        """Get unique values cnt.

        Be aware that this function uses approx_count_distinct and thus cannot return precise results

        Args:
            train: Dataset with train data.
            feats: Features names.

        Returns:
            Series.

        """
        warn_if_not_cached(train.data)

        sdf = train.data.select(feats)

        # TODO SPARK-LAMA: Do we really need this sampling?
        # if self.subsample:
        #     sdf = sdf.sample(withReplacement=False, fraction=self.subsample, seed=self.random_state)

        sdf = sdf.select([sf.approx_count_distinct(col).alias(col) for col in feats])
        result = sdf.collect()[0]

        uns = [result[col] for col in feats]
        return Series(uns, index=feats, dtype="int")

    def get_top_categories(self, train: SparkDataset, top_n: int = 5) -> List[str]:
        """Get top categories by importance.

        If feature importance is not defined,
        or feats has same importance - sort it by unique values counts.
        In second case init param ``ascending_by_cardinality``
        defines how - asc or desc.

        Args:
            train: Dataset with train data.
            top_n: Number of top categories.

        Returns:
            List.

        """
        if self.max_intersection_depth <= 1 or self.top_intersections <= 1:
            return []

        cats = get_columns_by_role(train, "Category")
        if len(cats) == 0:
            return []

        df = DataFrame({"importance": 0, "cardinality": 0}, index=cats)
        # importance if defined
        if self.feats_imp is not None:
            feats_imp = Series(self.feats_imp.get_features_score()).sort_values(ascending=False)
            df["importance"] = feats_imp[feats_imp.index.isin(cats)]
            df["importance"].fillna(-np.inf)

        # check for cardinality
        df["cardinality"] = self.get_uniques_cnt(train, cats)
        # sort
        df = df.sort_values(
            by=["importance", "cardinality"],
            ascending=[False, self.ascending_by_cardinality],
        )
        # get top n
        top = list(df.index[:top_n])

        return top


class SparkEmptyFeaturePipeline(SparkFeaturesPipeline):
    """
    This class creates pipeline with ``SparkNoOpTransformer``
    """

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
        """
        Returns ``SparkNoOpTransformer`` instance
        """
        return SparkNoOpTransformer(train.roles)


class SparkNoOpTransformer(SparkBaseTransformer, CommonPickleMLWritable, CommonPickleMLReadable):
    """
    This transformer does nothing, it just returns the input dataframe unchanged.
    """

    def __init__(self, roles: RolesDict):
        cols = list(roles.keys())
        super().__init__(input_cols=cols, output_cols=cols, input_roles=roles, output_roles=roles)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return dataset


class SparkPipelineModel(PipelineModel, SparkColumnsAndRoles):
    def __init__(self, stages: List[SparkBaseTransformer], input_roles: RolesDict, output_roles: RolesDict):
        super(SparkPipelineModel, self).__init__(stages)
        self.set(self.inputCols, list(input_roles.keys()))
        self.set(self.outputCols, list(output_roles.keys()))
        self.set(self.inputRoles, input_roles)
        self.set(self.outputRoles, output_roles)
