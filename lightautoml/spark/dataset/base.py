from copy import copy
from typing import Sequence, Any, Tuple, Union, Optional, NewType, List, cast, Dict

import pandas as pd

import pyspark
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F, Column
from pyspark.sql.session import SparkSession

from lightautoml.dataset.base import LAMLDataset, IntIdx, RowSlice, ColSlice, LAMLColumn, RolesDict
from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset, NpRoles
from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.tasks import Task

SparkDataFrame = NewType('SparkDataFrame', pyspark.sql.DataFrame)


class SparkDataset(LAMLDataset):
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "SparkDataset"

    def __init__(self, data: SparkDataFrame, roles: Optional[RolesDict], task: Optional[Task] = None,
                 **kwargs: Any):
        self._data = None
        super().__init__(data, None, roles, task, **kwargs)

    @property
    def spark_session(self):
        return SparkSession.getActiveSession()

    @property
    def data(self) -> SparkDataFrame:
        return self._data

    @data.setter
    def data(self, val: SparkDataFrame) -> None:
        self._data = val

    @property
    def features(self) -> List[str]:
        """Get list of features.

        Returns:
            list of features.

        """
        return [] if self.data is None else list(self.data.columns)

    @features.setter
    def features(self, val: None):
        """Ignore setting features.

        Args:
            val: ignored.

        """
        pass

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
        elif val:
            role = cast(ColumnRole, val)
            self._roles = dict(((x, role) for x in self.features))
        else:
            raise ValueError()

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        # TODO: we may call count here but it may lead to a huge calculation
        assert self.data.is_cached, "Shape should not be calculated if data is not cached"
        return self.data.count(), len(self.data.columns)

    def __repr__(self):
        return f"SparkDataset ({self.data})"

    def __getitem__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def __getitemf__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def __setitem__(self, k: str, val: Any):
        raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def _materialize_to_pandas(self) -> Tuple[pd.DataFrame, Dict[str, ColumnRole]]:
        sdf = self.data

        def expand_if_vec_or_arr(col, role) -> Tuple[List[Column], ColumnRole]:
            if not isinstance(role, NumericVectorOrArrayRole):
                return [col], role
            vrole = cast(NumericVectorOrArrayRole, role)

            def to_array(column):
                if vrole.is_vector:
                    return vector_to_array(column)
                return column

            arr = [
                to_array(F.col(col))[i].alias(vrole.feature_name_at(i))
                for i in range(vrole.size)
            ]

            return arr, NumericRole(dtype=vrole.dtype)

        arr_cols = (expand_if_vec_or_arr(c, self.roles[c]) for c in self.features)
        all_cols_and_roles = [(c, role) for c_arr, role in arr_cols for c in c_arr]
        all_cols = [scol for scol, _ in all_cols_and_roles]

        sdf = sdf.select(all_cols)
        data = sdf.toPandas()

        # we do it this way, because scol doesn't have a method to retrive name as a str
        all_roles = {c: r for c, (_, r) in zip(sdf.columns, all_cols_and_roles)}
        return pd.DataFrame(data=data.to_dict()), all_roles

    def set_data(self, data: SparkDataFrame, features: List[str],  roles: NpRoles = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `ignored, always None. just for same interface.
            roles: Dict with roles.

        """
        super().set_data(data, None, roles)

    def to_pandas(self) -> PandasDataset:
        data, roles = self._materialize_to_pandas()
        return PandasDataset(data=data, roles=roles, task=self.task)

    def to_numpy(self) -> NumpyDataset:
        data, roles = self._materialize_to_pandas()
        return NumpyDataset(data=data.to_numpy(), features=list(data.columns), roles=roles, task=self.task)

    @staticmethod
    def _hstack(datasets: Sequence[Any]) -> Any:
        # TODO: we need to use join to implement this operation
        # TODO: to perform join correctly we need to have unique id for each row
        # TODO: and optionally have bucketed tables (bucketed by the same way)
        raise NotImplementedError("It is not yet ready")

    @staticmethod
    def _get_cols(data, k: IntIdx) -> Any:
        raise NotImplementedError("It is not yet ready")
