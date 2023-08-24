from typing import List
from typing import Union

import numpy as np

from lightautoml.dataset.roles import Dtype
from lightautoml.dataset.roles import NumericRole


class NumericVectorOrArrayRole(NumericRole):
    """
    Role that describe numeric vector or numeric array.
    """

    _name = "NumericVectorOrArray"

    def __init__(
        self,
        size: int,
        element_col_name_template: Union[str, List[str]],
        dtype: Dtype = np.float32,
        force_input: bool = False,
        prob: bool = False,
        discretization: bool = False,
        is_vector: bool = True,
    ):
        """
        Args:
            size: number of elements in every vector in this column
            element_col_name_template: string template to produce name for each element in the vector
            when array-to-columns transformation is neccessary
            dtype: type of the vector's elements
            force_input: Select a feature for training,
              regardless of the selector results.
            prob: If input number is probability.
        """
        super().__init__(dtype, force_input, prob, discretization)
        self.size = size
        self.element_col_name_template = element_col_name_template
        self.is_vector = is_vector

    def feature_name_at(self, position: int) -> str:
        """
        produces a name for feature on ``position`` in the vector

        Args:
            position: position in the vector in range [0 .. size]

        Returns:
            feature name

        """
        assert 0 <= position < self.size

        if isinstance(self.element_col_name_template, str):
            return self.element_col_name_template.format(position)

        return self.element_col_name_template[position]
