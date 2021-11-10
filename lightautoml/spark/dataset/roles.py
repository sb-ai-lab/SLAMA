from lightautoml.dataset.roles import NumericRole, Dtype
import numpy as np


class NumericVectorRole(NumericRole):
    _name = "NumericVector"

    def __init__(
            self,
            size: int,
            element_col_name_template: str,
            dtype: Dtype = np.float32,
            force_input: bool = False,
            prob: bool = False,
            discretization: bool = False,
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
        self._size = size
        self._element_col_name_template = element_col_name_template

    @property
    def size(self):
        return self._size

    def feature_name_at(self, position: int) -> str:
        """
        produces a name for feature on ``position`` in the vector
        Args:
            position: position in the vector in range [0 .. size]

        Returns:
            feature name

        """
        assert 0 <= position < self.size
        return self._element_col_name_template.format(position)
