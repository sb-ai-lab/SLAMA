import logging.config
import sys

from examples_utils import BUCKET_NUMS
from examples_utils import get_dataset
from examples_utils import get_spark_session
from examples_utils import prepare_dataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import log_exec_timer
from sparklightautoml.utils import logging_config

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main(dataset_name: str, seed: int = 42):
    cv = 5
    dataset = get_dataset(dataset_name)
    persistence_manager = PlainCachePersistenceManager()

    with log_exec_timer("spark-lama training"):
        task = SparkTask(dataset.task_type)
        data = prepare_dataset(dataset, seed)

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(data, roles=dataset.roles, persistence_manager=persistence_manager)

        print("Inferred roles:")
        print(sdataset.roles)


if __name__ == "__main__":
    assert len(sys.argv) <= 2, "There may be no more than one argument"
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "lama_test_dataset"
    # if one uses bucketing based persistence manager,
    # the argument below number should be equal to what is set to 'bucket_nums' of the manager
    spark_sess = get_spark_session(BUCKET_NUMS)

    main(dataset_name)

    spark_sess.stop()
