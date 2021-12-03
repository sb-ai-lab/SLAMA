from collections import defaultdict
from itertools import chain, combinations
from typing import Optional, Sequence, List, Tuple, Dict, Union, cast

import numpy as np
from pandas import Series
from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame, Window
from pyspark.sql.functions import udf, array, monotonically_increasing_id
from pyspark.sql.types import FloatType, DoubleType, IntegerType
from sklearn.utils.murmurhash import murmurhash3_32

from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.categorical import categorical_check, encoding_check, oof_task_check, \
    multiclass_task_check

# FIXME SPARK-LAMA: np.nan in str representation is 'nan' while Spark's NaN is 'NaN'. It leads to different hashes.
# FIXME SPARK-LAMA: If udf is defined inside the class, it not works properly.
# "if murmurhash3_32 can be applied to a whole pandas Series, it would be better to make it via pandas_udf"
# https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749534669
murmurhash3_32_udf = F.udf(lambda value: murmurhash3_32(value.replace("NaN", "nan"), seed=42), SparkTypes.IntegerType())


class LabelEncoder(SparkTransformer):

    _ad_hoc_types_mapper = defaultdict(
        lambda: "string",
        {
            "bool": "boolean",
            "int": "int",
            "int8": "int",
            "int16": "int",
            "int32": "int",
            "int64": "int",
            "int128": "bigint",
            "int256": "bigint",
            "integer": "int",
            "uint8": "int",
            "uint16": "int",
            "uint32": "int",
            "uint64": "int",
            "uint128": "bigint",
            "uint256": "bigint",
            "longlong": "long",
            "ulonglong": "long",
            "float16": "float",
            "float": "float",
            "float32": "float",
            "float64": "double",
            "float128": "double"
        }
    )

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0

    def __init__(self, *args, **kwargs):
        self._output_role = CategoryRole(np.int32, label_encoded=True)

    def _fit(self, dataset: SparkDataset) -> "LabelEncoder":

        roles = dataset.roles

        # cached_dataset = dataset.data.cache()
        dataset.cache()

        self.dicts = {}
        for i in dataset.features:
            role = roles[i]

            # TODO: think what to do with this warning
            co = role.unknown

            # FIXME SPARK-LAMA: Possible OOM point
            # TODO SPARK-LAMA: Can be implemented without multiple groupby and thus shuffling using custom UDAF.
            # May be an alternative it there would be performance problems.
            # https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749539901
            vals = dataset.data \
                .groupBy(i).count() \
                .where(F.col("count") > co) \
                .orderBy(["count", i], ascending=[False, True]) \
                .select(i) \
                .toPandas()

            # FIXME SPARK-LAMA: Do we really need collecting this data? It is used in transform method and
            # it may be joined. I propose to keep this variable as a spark dataframe. Discuss?
            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals[i])

        # cached_dataset.unpersist()
        dataset.uncache()

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        df = dataset.data

        for i in dataset.features:

            # FIXME SPARK-LAMA: Dirty hot-fix
            # TODO SPARK-LAMA: It can be done easier with only one select and without withColumn but a single select.
            # https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749506973

            role = dataset.roles[i]

            df = df.withColumn(i, F.col(i).cast(self._ad_hoc_types_mapper[role.dtype.__name__]))

            if i in self.dicts:

                if len(self.dicts[i]) > 0:

                    labels = F.create_map([F.lit(x) for x in chain(*self.dicts[i].to_dict().items())])

                    if np.issubdtype(role.dtype, np.number):
                        df = df \
                            .withColumn(i, F.when(F.col(i).isNull(), np.nan)
                                            .otherwise(F.col(i))
                                        ) \
                            .withColumn(i, labels[F.col(i)])
                    else:
                        if None in self.dicts[i].index:
                            df = df \
                                .withColumn(i, F.when(F.col(i).isNull(), self.dicts[i][None])
                                                .otherwise(labels[F.col(i)])
                                            )
                        else:
                            df = df \
                                .withColumn(i, labels[F.col(i)])
                else:
                    df = df \
                        .withColumn(i, F.lit(self._fillna_val))

            df = df.fillna(self._fillna_val, subset=[i]) \
                .withColumn(i, F.col(i).cast(self._ad_hoc_types_mapper[self._output_role.dtype.__name__])) \
                .withColumnRenamed(i, f"{self._fname_prefix}__{i}")
                # FIXME SPARK-LAMA: Probably we have to write a converter numpy/python/pandas types => spark types?

        output: SparkDataset = dataset.empty()
        output.set_data(df, self.features, self._output_role)

        return output


class FreqEncoder(LabelEncoder):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"

    _fillna_val = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(np.float32)

    def _fit(self, dataset: SparkDataset) -> "FreqEncoder":

        cached_dataset = dataset.data.cache()

        self.dicts = {}
        for i in cached_dataset.columns:
            vals = cached_dataset \
                .groupBy(i).count() \
                .where(F.col("count") > 1) \
                .orderBy(["count", i], ascending=[False, True]) \
                .select([i, "count"]) \
                .toPandas()

            self.dicts[i] = vals.set_index(i)["count"]

        cached_dataset.unpersist()

        return self


class OrdinalEncoder(LabelEncoder):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "ord"

    _spark_numeric_types = [
        SparkTypes.ByteType,
        SparkTypes.ShortType,
        SparkTypes.IntegerType,
        SparkTypes.LongType,
        SparkTypes.FloatType,
        SparkTypes.DoubleType,
        SparkTypes.DecimalType
    ]

    _fillna_val = np.nan

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(np.float32)

    def _fit(self, dataset: SparkDataset) -> "OrdinalEncoder":

        roles = dataset.roles

        dataset.cache()
        cached_dataset = dataset.data

        self.dicts = {}
        for i in dataset.features:
            role = roles[i]

            if not type(cached_dataset.schema[i].dataType) in self._spark_numeric_types:

                co = role.unknown

                cnts = cached_dataset \
                    .groupBy(i).count() \
                    .where((F.col("count") > co) & F.col(i).isNotNull() & ~F.isnan(F.col(i))) \
                    .select(i) \
                    .toPandas()

                cnts = Series(cnts[i].astype(str).rank().values, index=cnts[i])
                self.dicts[i] = cnts.append(Series([cnts.shape[0] + 1], index=[np.nan])).drop_duplicates()

        dataset.uncache()

        return self


class CatIntersectstions(LabelEncoder):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "inter"

    def __init__(self,
                 intersections: Optional[Sequence[Sequence[str]]] = None,
                 max_depth: int = 2):

        super().__init__()
        self.intersections = intersections
        self.max_depth = max_depth

    @staticmethod
    def _make_category(df: SparkDataFrame, cols: Sequence[str]) -> SparkDataFrame:
        lit = F.lit("_")
        col_name = f"({'__'.join(cols)})"
        columns_for_concat = []
        for col in cols:
            columns_for_concat.append(F.col(col))
            columns_for_concat.append(lit)
        columns_for_concat = columns_for_concat[:-1]

        return df.withColumn(
            col_name,
            murmurhash3_32_udf(
                F.concat(*columns_for_concat)
            )
        )

    def _build_df(self, dataset: SparkDataset) -> SparkDataset:

        df = dataset.data

        roles = {}

        for comb in self.intersections:
            df = self._make_category(df, comb)
            roles[f"({'__'.join(comb)})"] = CategoryRole(
                object,
                unknown=max((dataset.roles[x].unknown for x in comb)),
                label_encoded=True,
            )

        df = df.select(
            *dataset.service_columns, *[f"({'__'.join(comb)})" for comb in self.intersections]
        )

        output = dataset.empty()
        output.set_data(df, df.columns, roles)

        return output

    def _fit(self, dataset: SparkDataset):

        if self.intersections is None:
            self.intersections = []
            for i in range(2, min(self.max_depth, len(dataset.features)) + 1):
                self.intersections.extend(list(combinations(dataset.features, i)))

        inter_dataset = self._build_df(dataset)
        return super()._fit(inter_dataset)

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        inter_dataset = self._build_df(dataset)
        return super().transform(inter_dataset)


class OHEEncoder(SparkTransformer):
    """
    Simple OneHotEncoder over label encoded categories.
    """

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        make_sparse: Optional[bool] = None,
        total_feats_cnt: Optional[int] = None,
        dtype: type = np.float32,
    ):
        """

        Args:
            make_sparse: Create sparse matrix.
            total_feats_cnt: Initial features number.
            dtype: Dtype of new features.

        """
        self.make_sparse = make_sparse
        self.total_feats_cnt = total_feats_cnt
        self.dtype = dtype

        if self.make_sparse is None:
            assert self.total_feats_cnt is not None, "Param total_feats_cnt should be defined if make_sparse is None"

        self._ohe_transformer_and_roles: Optional[Tuple[Transformer, Dict[str, ColumnRole]]] = None

    def _fit(self, dataset: SparkDataset):
        """Calc output shapes.

        Automatically do ohe in sparse form if approximate fill_rate < `0.2`.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """

        sdf = dataset.data
        temp_sdf = sdf.cache()
        maxs = [F.max(c).alias(f"max_{c}") for c in dataset.features]
        mins = [F.min(c).alias(f"min_{c}") for c in dataset.features]
        mm = temp_sdf.select(maxs + mins).collect()[0].asDict()

        self._features = [f"{self._fname_prefix}__{c}" for c in dataset.features]

        ohe = OneHotEncoder(inputCols=dataset.features, outputCols=self._features, handleInvalid="error")
        transformer = ohe.fit(temp_sdf)
        temp_sdf.unpersist()

        roles = {
            f"{self._fname_prefix}__{c}": NumericVectorOrArrayRole(
                size=mm[f"max_{c}"] - mm[f"min_{c}"] + 1,
                element_col_name_template=[
                    f"{self._fname_prefix}_{i}__{c}"
                    for i in np.arange(mm[f"min_{c}"], mm[f"max_{c}"] + 1)
                ]
            ) for c in dataset.features
        }

        self._ohe_transformer_and_roles = (transformer, roles)

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """

        sdf = dataset.data

        ohe, roles = self._ohe_transformer_and_roles

        # transform
        data = ohe.transform(sdf).select(*dataset.service_columns, *list(roles.keys()))

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, roles)

        return output


class TargetEncoder(SparkTransformer):

    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(self, alphas: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0]):

        self.alphas = alphas

    def fit(self, dataset: SparkDataset):
        super().fit_transform(dataset)

    def fit_transform(self,
                      dataset: SparkDataset,
                      target_column: str,
                      folds_column: str) -> SparkDataset:

        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        data = dataset.data
        self._features = [f"{self._fname_prefix}__{c}" for c in data.columns]

        self.encodings = []
        prior = data.agg(F.avg(F.col(target_column).cast("double"))).collect()[0][0]
        # folds priorss
        f_sum = data.groupBy(F.col(folds_column)).agg(F.sum(F.col(target_column).cast("double")).alias("sumd"), F.count(F.col(target_column).cast("double")).alias("countd"))
        tot_sum = data.agg(F.sum(F.col(target_column).cast("double")).alias("sumT")).collect()[0][0]
        tot_count = data.agg(F.count(F.col(target_column).cast("double")).alias("countT")).collect()[0][0]

        folds_prior = f_sum.withColumn("folds_prior", udf(lambda x, y: (tot_sum - x) / (tot_count - y), FloatType())(F.col("sumd"), F.col("countd")))
        # oof_feats = np.zeros(data.shape, dtype=np.float32)
        self.feats = data
        self.old_cols = data.columns
        self.old_cols.remove('_id')
        self.encs = {}

        for col_name in self.old_cols:

            if col_name == '_id':
                continue

            # calc folds stats
            enc_dim = data.agg(F.max(F.col(col_name))).collect()[0][0] + 1

            f_sum = data.groupBy(F.col(folds_column), F.col(col_name)).agg(F.sum(F.col(target_column).cast("double")).alias("sumd"), F.count(F.col(target_column).cast("double")).alias("countd"))

            # calc total stats
            t_sum = data.groupBy(F.col(col_name)).agg(F.sum(F.col(target_column).cast("double")).alias("sumt"), F.count(F.col(target_column).cast("double")).alias("countt"))

            # calc oof stats
            f_sum = f_sum.join(t_sum, col_name)
            oof_sum = f_sum.withColumn("oof_sum", F.col("sumt") - F.col(("sumd"))).withColumn("oof_count", F.col("countt") - F.col(("countd")))
            oof_sum_joined = oof_sum.join(data, [col_name, folds_column]).join(folds_prior, folds_column)
            diff = {}
            udf_diff = udf(lambda x, y: float(-(y * np.log(x) + (1 - y) * (np.log(1 - x)))), DoubleType()) if dataset.task.name == "binary" \
                  else udf(lambda x, y: float(((y - x) ** 2)), DoubleType())
            # calc candidates alpha
            for a in self.alphas:
                udf_a = udf(lambda os, fp, oc: (os + a * fp) / (oc + a), DoubleType())
                alp_colname = f"walpha{a}".replace(".", "d").replace(",", "d")
                dif_colname = f"{alp_colname}Diff"
                oof_sum_joined = oof_sum_joined.withColumn(alp_colname, udf_a(F.col("oof_sum").cast("double"), F.col("folds_prior").cast("double"), F.col("oof_count").cast("double")))
                oof_sum_joined = oof_sum_joined.withColumn(dif_colname, udf_diff(F.col(alp_colname).cast("double"), F.col(target_column).cast("double")))
                diff[a] = oof_sum_joined.agg(F.avg(F.col(alp_colname + "Diff").cast("double"))).collect()[0][0]
            a_opt = min(diff, key=diff.get)
            out_col = f"walpha{a_opt}".replace(".", "d").replace(",", "d")

            w = Window.orderBy(monotonically_increasing_id())

            self.feats = self.feats.withColumn("columnindex", F.row_number().over(w))
            oof_sum_joined = oof_sum_joined.withColumn("columnindex", F.row_number().over(w))

            self.feats = self.feats.alias('a').join(oof_sum_joined.withColumnRenamed(out_col, self._fname_prefix + "__" +col_name).alias('b'),
                                                     self.feats.columnindex == oof_sum_joined.columnindex, 'inner').select(
                [F.col('a.' + xx) for xx in self.feats.columns] + [F.col('b.{}'.format(self._fname_prefix + "__" +col_name))]).drop(self.feats.columnindex)

            # calc best encoding
            enc = t_sum.withColumn(f"enc_{col_name}", udf(lambda tot_sum, tot_count: float((tot_sum + a_opt * prior) / (tot_count + a_opt)), FloatType())(F.col("sumt"), F.col("countt"))).select(f"enc_{col_name}")
            enc = list(map(lambda x: x[0], enc.collect()))
            self.encs[col_name] = enc

        output = dataset.empty()
        self.output_role = NumericRole(np.float32, prob=True)
        self.feats = self.feats.drop(*self.old_cols)
        output.set_data(self.feats, self.features, self.output_role)

        return output

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        # checks here
        super().transform(dataset)

        data = dataset.data

        # transform
        for c in self.old_columns:
            data = data.withColumn(c, data[c].cast("double"))
            data = data.replace(to_replace=dict(enumerate(self.encs[c])), subset=c)

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, self.output_role)

        return output


from lightautoml.transformers.categorical import TargetEncoder as LAMATargetEncoder
from lightautoml.dataset.np_pd_dataset import NumpyDataset, PandasDataset
class MockTargetEncoder(LAMATargetEncoder, SparkTransformer):

    def _convert_to_lama(self, dataset: SparkDataset) -> Union[NumpyDataset, PandasDataset]:
        return dataset.to_numpy()

    def _convert_to_spark(self, dataset: Union[NumpyDataset, PandasDataset], spark) -> SparkDataset:
        return SparkDataset.from_lama(dataset, spark)

    def fit(self, dataset: SparkDataset) -> SparkDataset:
        lama_ds = self._convert_to_lama(dataset)
        fitted_ds = LAMATargetEncoder.fit(self, lama_ds)
        converted_ds = self._convert_to_spark(fitted_ds, spark=dataset.data.sql_ctx.sparkSession)
        return converted_ds

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        return self._convert_to_spark(
            LAMATargetEncoder.transform(
                self,
                self._convert_to_lama(dataset)
            ),
            spark=dataset.data.sql_ctx.sparkSession
        )

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        return self._convert_to_spark(
            LAMATargetEncoder.fit_transform(
                self,
                self._convert_to_lama(dataset)
            ),
            spark=dataset.data.sql_ctx.sparkSession
        )


class MultiClassTargetEncoder(SparkTransformer):

    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self, alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0)):
        self.alphas = alphas

    #TODO
    @staticmethod
    def score_func(candidates: np.ndarray, target: np.ndarray) -> int:

        target = target[:, np.newaxis, np.newaxis]
        scores = -np.log(np.take_along_axis(candidates, target, axis=1)).mean(axis=0)[0]
        idx = scores.argmin()

        return idx

    def fit_transform(self,
                      dataset: SparkDataset,
                      target_column: str,
                      folds_column: str) -> SparkDataset:

        # set transformer names and add checks
        super().fit(dataset)

        data = dataset.data

        self.encodings = []
        prior = data.groupBy(target_column).agg(F.count(F.col(target_column).cast("double"))/data.count()).collect()
        prior = list(map(lambda x: x[1], prior))

        f_sum = data.groupBy(target_column, folds_column).agg(F.count(F.col(target_column).cast("double")).alias("sumd"))
        f_count = data.groupBy(folds_column).agg(F.count(F.col(target_column).cast("double")).alias("countd"))
        f_sum = f_sum.join(f_count, folds_column)
        tot_sum = data.groupBy(target_column).agg(F.count(F.col(target_column).cast("double")).alias("sumT"))
        f_sum = f_sum.join(tot_sum, target_column)
        tot_count = data.agg(F.count(F.col(target_column).cast("double")).alias("countT")).collect()[0][0]

        folds_prior = f_sum.withColumn("folds_prior", udf(lambda x, y, z: (z - x) / (tot_count - y), FloatType())(F.col("sumd"), F.col("countd"), F.col("sumT")))
        self.feats = data
        self.encs = {}
        self.n_classes = data.agg(F.max(target_column)).collect()[0][0] + 1
        self.old_columns = data.columns
        self.old_columns.remove('_id')
        self._features = []
        for i in dataset.features:
            for j in range(self.n_classes):
                self._features.append("{0}_{1}__{2}".format("multioof", j, i))

        for col_name in self.old_columns:

            f_sum = data.groupBy(col_name, target_column, folds_column).agg(F.count(F.col(target_column).cast("double")).alias("sumd"))
            f_count = data.groupBy(col_name, folds_column).agg(F.count(F.col(target_column).cast("double")).alias("countd"))
            t_sum = data.groupBy(col_name, target_column).agg(F.count(F.col(target_column).cast("double")).alias("sumt"))
            t_count = data.groupBy(col_name).agg(F.count(F.col(target_column).cast("double")).alias("countt"))
            f_sum = f_sum.join(f_count, [col_name, folds_column]).join(t_sum, [col_name, target_column]).join(t_count, col_name)

            oof_sum = f_sum.withColumn("oof_sum", F.col("sumt") - F.col(("sumd"))).withColumn("oof_count", F.col("countt") - F.col(("countd")))
            oof_sum_joined = oof_sum.join(data[col_name, target_column, folds_column], [col_name, target_column, folds_column]).join(folds_prior[folds_column, target_column, "folds_prior"], [folds_column, target_column])
            diff = {}
            udf_diff = udf(lambda x: float(-(np.log(x))), DoubleType())

            for a in self.alphas:
                udf_a = udf(lambda os, fp, oc: (os + a * fp) / (oc + a), DoubleType())
                alp_colname = f"walpha{a}".replace(".", "d").replace(",", "d")
                dif_colname = f"{alp_colname}Diff"
                oof_sum_joined = oof_sum_joined.withColumn(alp_colname, udf_a(F.col("oof_sum").cast("double"), F.col("folds_prior").cast("double"), F.col("oof_count").cast("double")))
                oof_sum_joined = oof_sum_joined.withColumn(dif_colname, udf_diff(F.col(alp_colname).cast("double")))
                diff[a] = oof_sum_joined.agg(F.avg(F.col(alp_colname + "Diff").cast("double"))).collect()[0][0]
            a_opt = min(diff, key=diff.get)
            out_col = f"walpha{a_opt}".replace(".", "d").replace(",", "d")

            w = Window.orderBy(monotonically_increasing_id())

            self.feats = self.feats.withColumn("columnindex", F.row_number().over(w))
            oof_sum_joined = oof_sum_joined.withColumn("columnindex", F.row_number().over(w))

            self.feats = self.feats.alias('a').join(oof_sum_joined.withColumnRenamed(out_col, self._fname_prefix + "__" +col_name).alias('b'),
                                                     self.feats.columnindex == oof_sum_joined.columnindex, 'inner').select(
                [F.col('a.' + xx) for xx in self.feats.columns] + [F.col('b.{}'.format(self._fname_prefix + "__" +col_name))]).drop(self.feats.columnindex)

            # calc best encoding
            enc_list = []
            for i in range(self.n_classes):
                enc = f_sum.withColumn(f"enc_{col_name}", udf(lambda tot_sum, tot_count: float((tot_sum + a_opt * prior[i]) / (tot_count + a_opt)), FloatType())(F.col("sumt"), F.col("countt"))).select(f"enc_{col_name}")
                enc = list(map(lambda x: x[0], enc.collect()))
                enc_list.append(enc)
            sums = [sum(x) for x in zip(*enc_list)]
            for i in range(self.n_classes):
                self.encs[self._fname_prefix + "_" + str(i) + "__" + col_name ] = [enc_list[i][j]/sums[j] for j in range(len(enc_list[i]))]

        self.output_role = NumericRole(np.float32, prob=True)
        return self.transform(dataset)

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        # checks here
        super().transform(dataset)

        data = dataset.data
        dc = self.old_columns
        # transform
        for c in dc:
            data = data.withColumn(c, data[c].cast("int"))
            for i in range(self.n_classes):
                col = self.encs[self._fname_prefix + "_" + str(i) + "__" + c]
                data = data.withColumn(self._fname_prefix + "_" + str(i) + "__" + c, udf(lambda x: col[x], DoubleType())(c))
            data = data.drop(c)

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, self.output_role)

        return output
