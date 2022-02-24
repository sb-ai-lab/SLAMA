from typing import List

from lightautoml.dataset.base import RolesDict


class InputFeaturesAndRoles:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_roles = None

    @property
    def input_features(self) -> List[str]:
        return list(self.input_roles.keys())

    @property
    def input_roles(self) -> RolesDict:
        return self._input_roles

    @input_roles.setter
    def input_roles(self, roles: RolesDict):
        assert roles is None or isinstance(roles, dict)
        self._input_roles = roles


class OutputFeaturesAndRoles:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_roles = None

    @property
    def output_features(self) -> List[str]:
        return list(self.output_roles.keys())

    @property
    def output_roles(self) -> RolesDict:
        return self._output_roles
