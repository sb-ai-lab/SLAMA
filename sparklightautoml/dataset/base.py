import functools
import logging
import os
import pickle
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Any, Tuple, Union, Optional, List, cast, Dict, Set, Callable

import pandas as pd
from lightautoml.dataset.base import (
    LAMLDataset,
    IntIdx,
    RowSlice,
    ColSlice,
    LAMLColumn,
    RolesDict,
    valid_array_attributes,
    array_attr_roles,
)
from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset, NpRoles
from lightautoml.dataset.roles import ColumnRole, NumericRole, DropRole
from lightautoml.tasks import Task
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as sf, Column
from pyspark.sql.session import SparkSession

from sparklightautoml import VALIDATION_COLUMN
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.utils import SparkDataFrame, create_directory

logger = logging.getLogger(__name__)

Dependency = Union[str, 'SparkDataset', 'Unpersistable', Callable]
DepIdentifable = Union[str, 'SparkDataset']


class PersistenceLevel(Enum):
    """
        Used for signaling types of persistence points encountered during AutoML process.
    """
    READER = 0
    REGULAR = 1
    CHECKPOINT = 2


class Unpersistable(ABC):
    """
        Interface to provide for external entities to unpersist dataframes and files stored
        by the entity that implements this interface
    """
    def unpersist(self):
        ...


class SparkDataset(LAMLDataset, Unpersistable):
    """
    Implements a dataset that uses a ``pyspark.sql.DataFrame`` internally,
    stores some internal state (features, roles, ...) and provide methods to work with dataset.
    """

    @staticmethod
    def _get_rows(data, k: IntIdx) -> Any:
        raise NotImplementedError("This method is not supported")

    @staticmethod
    def _set_col(data: Any, k: int, val: Any):
        raise NotImplementedError("This method is not supported")

    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "SparkDataset"

    ID_COLUMN = "_id"

    @classmethod
    def load(cls,
             path: str,
             file_format: str = 'parquet',
             options: Optional[Dict[str, Any]] = None,
             persistence_manager: Optional['PersistenceManager'] = None) -> 'SparkDataset':
        metadata_file_path = os.path.join(path, f"metadata.{file_format}")
        file_path = os.path.join(path, f"data.{file_format}")
        options = options or dict()
        spark = SparkSession.getActiveSession()

        # reading metadata
        metadata_df = spark.read.format(file_format).options(**options).load(metadata_file_path)
        metadata = pickle.loads(metadata_df.select('metadata').first().asDict()['metadata'])

        # reading data
        data_df = spark.read.format(file_format).options(**options).load(file_path)
        name_fixed_cols = (sf.col(c).alias(c.replace('[', '(').replace(']', ')')) for c in data_df.columns)
        data_df = data_df.select(*name_fixed_cols)

        return SparkDataset(data=data_df, persistence_manager=persistence_manager, **metadata)

    # TODO: SLAMA - implement filling dependencies
    @classmethod
    def concatenate(
            cls,
            datasets: Sequence["SparkDataset"],
            name: Optional[str] = None,
            extra_dependencies: Optional[List[Dependency]] = None
    ) -> "SparkDataset":
        """
        Concat multiple datasets by joining their internal ``pyspark.sql.DataFrame``
        using inner join on special hidden '_id' column
        Args:
            datasets: spark datasets to be joined

        Returns:
            a joined dataset, containing features (and columns too) from all datasets
            except containing only one _id column
        """
        assert len(datasets) > 0, "Cannot join an empty list of datasets"

        if len(datasets) == 1:
            return datasets[0]

        if any(not d.bucketized for d in datasets):
            warnings.warn(
                f"NOT bucketized datasets are requested to be joined. It may severely affect performance",
                RuntimeWarning
            )

        # we should join datasets only with unique features
        features = [feat for ds in datasets for feat in ds.features]
        feat_counter = Counter(features)

        assert all(count == 1 for el, count in feat_counter.items()), \
            f"Different datasets being joined contain columns with the same names: {feat_counter}"

        roles = {col: role for ds in datasets for col, role in ds.roles.items()}

        except_cols = [c for c in datasets[0].service_columns if c != SparkDataset.ID_COLUMN]
        concatenated_sdf = functools.reduce(
            lambda acc, sdf: acc.join(sdf.drop(*except_cols), on=cls.ID_COLUMN, how='left'),
            (d.data for d in datasets)
        )

        output = datasets[0].empty()
        output.set_data(concatenated_sdf, features, roles, dependencies=[*datasets, *(extra_dependencies or [])], name=name)

        return output

    def __init__(self,
                 data: SparkDataFrame,
                 roles: Optional[RolesDict],
                 persistence_manager: Optional['PersistenceManager'] = None,
                 task: Optional[Task] = None,
                 bucketized: bool = False,
                 dependencies: Optional[List[Dependency]] = None,
                 name: Optional[str] = None,
                 target: Optional[str] = None,
                 folds: Optional[str] = None,
                 **kwargs: Any):
        self._validate_dataframe(data)

        roles = roles if roles else dict()

        # currently only target is supported
        # adapted from PandasDataset
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    roles[f] = DropRole()

        self._data = None
        self._bucketized = bucketized
        self._roles = None
        self._uid = str(uuid.uuid4())
        self._persistence_manager = persistence_manager
        self._dependencies = dependencies
        self._frozen = False
        self._name = name
        self._is_persisted = False
        self._target_column = target
        self._folds_column = folds
        # columns that can be transferred intact across all transformations
        # in the pipeline
        self._service_columns: Set[str] = {self.ID_COLUMN, target, folds, VALIDATION_COLUMN}

        super().__init__(data, list(roles.keys()), roles, task, **kwargs)

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def spark_session(self) -> SparkSession:
        return self.data.sql_ctx.sparkSession

    @property
    def data(self) -> SparkDataFrame:
        return self._data

    @data.setter
    def data(self, val: SparkDataFrame) -> None:
        self._data = val

    @property
    def dependencies(self) -> Optional[List[Dependency]]:
        return self._dependencies

    @property
    def features(self) -> List[str]:
        """Get list of features.

        Returns:
            list of features.

        """
        return self._features

    @features.setter
    def features(self, val: List[str]):
        """Ignore setting features.

        Args:
            val: ignored.

        """
        diff = set(val).difference(self.data.columns)
        assert len(diff) == 0, f"Not all roles have features in the dataset. Absent features: {diff}."
        self._features = copy(val)

    @property
    def roles(self) -> RolesDict:
        """Roles dict."""
        return copy(self._roles)

    @roles.setter
    def roles(self, val: NpRoles):
        """Define how to set roles.

        Args:
            val: Roles.

        Note:
            There is different behavior for different type of val parameter:

                - `List` - should be same len as ``data.shape[1]``.
                - `None` - automatic set ``NumericRole(np.float32)``.
                - ``ColumnRole`` - single role for all.
                - ``dict``.

        """
        if type(val) is dict:
            self._roles = dict(((x, val[x]) for x in self.features))
        elif type(val) is list:
            self._roles = dict(zip(self.features, val))
            diff = set(self._roles.keys()).difference(self.data.columns)
            assert len(diff) == 0, f"Not all roles have features in the dataset. Absent features: {diff}."

        elif val:
            role = cast(ColumnRole, val)
            self._roles = dict(((x, role) for x in self.features))
        else:
            raise ValueError()

        diff = set(self._roles.keys()).difference(self.data.columns)
        assert len(diff) == 0, f"Not all roles have features in the dataset. Absent features: {diff}."

    @property
    def bucketized(self) -> bool:
        return self._bucketized

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        return -1, len(self.features)
        # with JobGroup("sparkdataset.shape", f"Finding count for dataset (uid={self.uid}, name={self.name})"):
        #     warn_if_not_cached(self.data)
        #     return self.data.count(), len(self.features)

    @property
    def target_column(self) -> Optional[str]:
        return self._target_column

    @property
    def folds_column(self) -> Optional[str]:
        return self._folds_column

    @property
    def service_columns(self) -> List[str]:
        return [sc for sc in self._service_columns if sc in self.data.columns]

    @property
    def num_folds(self) -> int:
        return self.data.select(self.folds_column).distinct().count()

    @property
    def persistence_manager(self) -> 'PersistenceManager':
        return self._persistence_manager

    def __repr__(self):
        return f"SparkDataset ({self.data})"

    def __getitem__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        rslice, clice = k

        if isinstance(clice, str):
            clice = [clice]

        assert all(c in self.features for c in clice), (
            f"Not all columns presented in the dataset.\n" f"Presented: {self.features}\n" f"Asked for: {clice}"
        )

        present_svc_cols = [c for c in self.service_columns]
        sdf = cast(SparkDataFrame, self.data.select(*present_svc_cols, *clice))
        roles = {c: self.roles[c] for c in clice}

        output = self.empty()
        output.set_data(sdf, clice, roles, name=self.name)

        return output
        # raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def __setitem__(self, k: str, val: Any):
        raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SparkDataset) and o.uid == self.uid

    def __hash__(self) -> int:
        return hash(self.uid)

    def _validate_dataframe(self, sdf: SparkDataFrame) -> None:
        assert (
            self.ID_COLUMN in sdf.columns
        ), f"No special unique row id column (the column name: {self.ID_COLUMN}) in the spark dataframe"
        # assert kwargs["target"] in data.columns, \
        #     f"No target column (the column name: {kwargs['target']}) in the spark dataframe"

    def _materialize_to_pandas(
        self,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Dict[str, ColumnRole]]:
        sdf = self.data

        def expand_if_vec_or_arr(col, role) -> Tuple[List[Column], ColumnRole]:
            if not isinstance(role, NumericVectorOrArrayRole):
                return [col], role
            vrole = cast(NumericVectorOrArrayRole, role)

            def to_array(column):
                if vrole.is_vector:
                    return vector_to_array(column)
                return column

            arr = [to_array(sf.col(col))[i].alias(vrole.feature_name_at(i)) for i in range(vrole.size)]

            return arr, NumericRole(dtype=vrole.dtype)

        arr_cols = (expand_if_vec_or_arr(c, self.roles[c]) for c in self.features)
        all_cols_and_roles = {c: role for c_arr, role in arr_cols for c in c_arr}
        all_cols = [scol for scol, _ in all_cols_and_roles.items()]

        if self.target_column is not None:
            all_cols.append(self.target_column)

        if self.folds_column is not None:
            all_cols.append(self.folds_column)

        sdf = sdf.orderBy(SparkDataset.ID_COLUMN).select(*all_cols)
        all_roles = {c: all_cols_and_roles[c] for c in sdf.columns if c not in self.service_columns}

        data = sdf.toPandas()

        df = pd.DataFrame(data=data.to_dict())

        if self.target_column is not None:
            target_series = df[self.target_column]
            df = df.drop(self.target_column, axis=1)
        else:
            target_series = None

        if self.folds_column is not None:
            folds_series = df[self.folds_column]
            df = df.drop(self.folds_column, axis=1)
        else:
            folds_series = None

        return df, target_series, folds_series, all_roles

    def _initialize(self, task: Optional[Task], **kwargs: Any):
        super()._initialize(task, **kwargs)
        self._dependencies = None

    def empty(self) -> "SparkDataset":
        dataset = cast(SparkDataset, super().empty())
        dataset._dependencies = [self]
        dataset._uid = str(uuid.uuid4())
        dataset._frozen = False
        return dataset

    def set_data(
            self,
            data: SparkDataFrame,
            features: List[str],
            roles: NpRoles = None,
            persistence_manager: Optional['PersistenceManager'] = None,
            dependencies: Optional[List[Dependency]] = None,
            uid: Optional[str] = None,
            name: Optional[str] = None,
            frozen: bool = False
    ):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `ignored, always None. just for same interface.
            roles: Dict with roles.
        """
        self._validate_dataframe(data)
        super().set_data(data, features, roles)
        self._persistence_manager = persistence_manager or self._persistence_manager
        self._dependencies = dependencies if dependencies is not None else self._dependencies
        self._uid = uid or self._uid
        self._name = name or self._name
        self._frozen = frozen

    def persist(self, level: Optional[PersistenceLevel] = None, force: bool=False) -> 'SparkDataset':
        """
        Materializes current Spark DataFrame and unpersists all its dependencies
        Args:
            level:

        Returns:
            a new SparkDataset that is persisted and materialized
        """
        # if self._is_persisted:
        #     return self

        assert self.persistence_manager, "Cannot persist when persistence_manager is None"

        logger.debug(f"Persisting SparkDataset (uid={self.uid}, name={self.name})")
        level = level if level is not None else PersistenceLevel.REGULAR

        if force:
            ds = self.empty()
            ds.set_data(self.data, self.features, self.roles)
            persisted_dataset = self.persistence_manager.persist(ds, level).to_dataset()
        else:
            persisted_dataset = self.persistence_manager.persist(self, level).to_dataset()

        self._unpersist_dependencies()
        self._is_persisted = True

        return persisted_dataset

    def unpersist(self):
        """
        Unpersists current dataframe if it is persisted and all its dependencies.
        It does nothing if the dataset is frozen
        """
        assert self.persistence_manager, "Cannot unpersist when persistence_manager is None"

        if self.frozen:
            logger.debug(f"Cannot unpersist frozen SparkDataset (uid={self.uid}, name={self.name})")
            return

        logger.debug(f"Unpersisting SparkDataset (uid={self.uid}, name={self.name})")

        self.persistence_manager.unpersist(self.uid)
        self._unpersist_dependencies()

        # if self._is_persisted:
        #     self.persistence_manager.unpersist(self.uid)
        # else:
        #     self._unpersist_dependencies()

    def _unpersist_dependencies(self):
        for dep in (self.dependencies or []):
            if isinstance(dep, str):
                self.persistence_manager.unpersist(dep)
            elif isinstance(dep, Unpersistable):
                dep.unpersist()
            else:
                dep()

    @property
    def frozen(self) -> bool:
        return self._frozen

    @frozen.setter
    def frozen(self, val: bool):
        self._frozen = val

    def freeze(self) -> 'SparkDataset':
        ds = self.empty()
        ds.set_data(self.data, self.features, self.roles, frozen=True)
        return ds

    def save(self, path: str, save_mode: str = 'error', file_format: str = 'parquet', options: Optional[Dict[str, Any]] = None):
        metadata_file_path = os.path.join(path, f"metadata.{file_format}")
        file_path = os.path.join(path, f"data.{file_format}")
        options = options or dict()

        # prepare metadata of the dataset
        metadata = {
            "name": self.name,
            "roles": self.roles,
            "task": self.task,
            "target": self.target_column,
            "folds": self.folds_column,
        }
        metadata_str = pickle.dumps(metadata)
        metadata_df = self.spark_session.createDataFrame([{"metadata": metadata_str}])

        # create directory that will store data and metadata as separate files of dataframes
        create_directory(path, spark=self.spark_session, exists_ok=(save_mode in ['overwrite', 'append']))

        # writing dataframes
        metadata_df.write.format(file_format).mode(save_mode).options(**options).save(metadata_file_path)
        # fix name of columns: parquet cannot have columns with '(' or ')' in the name
        name_fixed_cols = (sf.col(c).alias(c.replace('(', '[').replace(')', ']')) for c in self.data.columns)
        self.data.select(*name_fixed_cols).write.format(file_format).mode(save_mode).options(**options).save(file_path)

    def to_pandas(self) -> PandasDataset:
        data, target_data, folds_data, roles = self._materialize_to_pandas()

        task = Task(self.task.name) if self.task else None
        kwargs = dict()
        if target_data is not None:
            kwargs["target"] = target_data
        if folds_data is not None:
            kwargs["folds"] = folds_data
        pds = PandasDataset(data=data, roles=roles, task=task, **kwargs)

        return pds

    def to_numpy(self) -> NumpyDataset:
        data, target_data, folds_data, roles = self._materialize_to_pandas()

        try:
            target = self.target
            if isinstance(target, pd.Series):
                target = target.to_numpy()
            elif isinstance(target, SparkDataFrame):
                target = target.toPandas().to_numpy()
        except AttributeError:
            target = None

        try:
            folds = self.folds
            if isinstance(folds, pd.Series):
                folds = folds.to_numpy()
            elif isinstance(folds, SparkDataFrame):
                folds = folds.toPandas().to_numpy()
        except AttributeError:
            folds = None

        return NumpyDataset(
            data=data.to_numpy(), features=list(data.columns), roles=roles, task=self.task, target=target, folds=folds
        )

    @staticmethod
    def _hstack(datasets: Sequence[Any]) -> Any:
        raise NotImplementedError("Unsupported operation for this dataset type")

    @staticmethod
    def _get_cols(data, k: IntIdx) -> Any:
        raise NotImplementedError("Unsupported operation for this dataset type")

    @staticmethod
    def from_dataset(dataset: "LAMLDataset") -> "LAMLDataset":
        assert isinstance(dataset, SparkDataset), "Can only convert from SparkDataset"
        return dataset


@dataclass(frozen=True)
class PersistableDataFrame:
    sdf: SparkDataFrame
    uid: str
    callback: Optional[Callable] = None
    base_dataset: Optional[SparkDataset] = None
    custom_name: Optional[str] = None

    def to_dataset(self) -> SparkDataset:
        assert self.base_dataset is not None
        ds = self.base_dataset.empty()
        ds.set_data(
            self.sdf,
            self.base_dataset.features,
            self.base_dataset.roles,
            dependencies=list(self.base_dataset.dependencies or []),
            uid=self.uid,
            name=self.base_dataset.name
        )
        return ds

    @property
    def name(self) -> Optional[str]:
        ds_name = self.base_dataset.name if self.base_dataset is not None else None
        return self.custom_name or ds_name


class PersistenceManager(ABC):
    """
        Base interface of an entity responsible for caching and storing intermediate results somewhere.
    """
    @staticmethod
    def to_persistable_dataframe(dataset: SparkDataset) -> PersistableDataFrame:
        # we intentially create new uid to use to distinguish a persisted and unpersisted dataset
        return PersistableDataFrame(dataset.data, uid=dataset.uid, base_dataset=dataset)

    @property
    @abstractmethod
    def uid(self) -> str:
        ...

    @property
    @abstractmethod
    def children(self) -> List['PersistenceManager']:
        ...

    @property
    @abstractmethod
    def datasets(self) -> List[SparkDataset]:
        ...

    @property
    @abstractmethod
    def all_datasets(self) -> List[SparkDataset]:
        """
        Returns:
            all persisted datasets including persisted with children contexts
        """
        ...

    @abstractmethod
    def persist(self,
                dataset: Union[SparkDataset, PersistableDataFrame],
                level: PersistenceLevel = PersistenceLevel.REGULAR) -> PersistableDataFrame:
        ...

    @abstractmethod
    def unpersist(self, uid: str):
        ...

    @abstractmethod
    def unpersist_all(self):
        ...

    @abstractmethod
    def unpersist_children(self):
        ...

    @abstractmethod
    def child(self) -> 'PersistenceManager':
        ...

    @abstractmethod
    def remove_child(self, child: Union['PersistenceManager', str]):
        ...

    @abstractmethod
    def is_persisted(self, pdf: PersistableDataFrame) -> bool:
        ...
