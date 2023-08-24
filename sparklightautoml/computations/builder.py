from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.base import ComputationsSettings
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.computations.sequential import SequentialComputationsManager


# either named profile and parallelism or parallelism settings or factory
AutoMLComputationsSettings = Union[Tuple[str, int], Dict[str, Any], "ComputationsManagerFactory"]


class ComputationsManagerFactory:
    def __init__(self, computations_settings: Optional[Union[Tuple[str, int], Dict[str, Any]]] = None):
        super(ComputationsManagerFactory, self).__init__()
        computations_settings = computations_settings or ("no_parallelism", -1)

        if isinstance(computations_settings, Tuple):
            mode, parallelism = computations_settings
            self._computations_settings = build_named_parallelism_settings(mode, parallelism)
        else:
            self._computations_settings = computations_settings

        self._ml_pipelines_params = self._computations_settings.get("ml_pipelines", None)
        self._ml_algos_params = self._computations_settings.get("ml_algos", None)
        self._selector_params = self._computations_settings.get("selector", None)
        self._tuner_params = self._computations_settings.get("tuner", None)
        self._linear_l2_params = self._computations_settings.get("linear_l2", None)
        self._lgb_params = self._computations_settings.get("lgb", None)

    def get_ml_pipelines_manager(self) -> "ComputationsManager":
        return build_computations_manager(self._ml_pipelines_params)

    def get_ml_algo_manager(self) -> "ComputationsManager":
        return build_computations_manager(self._ml_algos_params)

    def get_selector_manager(self) -> "ComputationsManager":
        return build_computations_manager(self._selector_params)

    def get_tuning_manager(self) -> "ComputationsManager":
        return build_computations_manager(self._tuner_params)

    def get_lgb_manager(self) -> "ComputationsManager":
        return build_computations_manager(self._lgb_params)

    def get_linear_manager(self) -> "ComputationsManager":
        return build_computations_manager(self._linear_l2_params)


def build_named_parallelism_settings(config_name: str, parallelism: int):
    intra_mlpipe_parallelim = {
        "ml_pipelines": {"parallelism": 1},
        "ml_algos": {"parallelism": 1},
        "selector": {"parallelism": parallelism},
        "tuner": {"parallelism": parallelism},
        "linear_l2": {"parallelism": parallelism},
        "lgb": {"parallelism": parallelism, "use_location_prefs_mode": False},
    }

    parallelism_config = {
        "no_parallelism": {},
        "parallelism": intra_mlpipe_parallelim,
        "intra_mlpipe_parallelism": intra_mlpipe_parallelim,
        "intra_mlpipe_parallelism_with_location_prefs_mode": {
            "ml_pipelines": {"parallelism": 1},
            "ml_algos": {"parallelism": 1},
            "selector": {"parallelism": parallelism},
            "tuner": {"parallelism": parallelism},
            "linear_l2": {"parallelism": parallelism},
            "lgb": {"parallelism": parallelism, "use_location_prefs_mode": True},
        },
        "mlpipe_level_parallelism": {
            "ml_pipelines": {"parallelism": parallelism},
            "ml_algos": {"parallelism": 1},
            "selector": {"parallelism": 1},
            "tuner": {"parallelism": 1},
            "linear_l2": {"parallelism": 1},
            "lgb": {"parallelism": 1},
        },
    }

    assert config_name in parallelism_config, (
        f"Not supported parallelism mode: {config_name}. "
        f"Only the following ones are supoorted at the moment: {list(parallelism_config.keys())}"
    )

    return parallelism_config[config_name]


def build_computations_manager(computations_settings: Optional[ComputationsSettings] = None) -> "ComputationsManager":
    if computations_settings is not None and isinstance(computations_settings, ComputationsManager):
        computations_manager = computations_settings
    elif computations_settings is not None:
        assert isinstance(computations_settings, dict)
        parallelism = int(computations_settings.get("parallelism", "1"))
        use_location_prefs_mode = computations_settings.get("use_location_prefs_mode", False)
        computations_manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    else:
        computations_manager = SequentialComputationsManager()

    return computations_manager
