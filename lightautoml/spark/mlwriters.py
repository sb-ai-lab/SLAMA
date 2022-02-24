import logging
from pyspark.ml.util import MLReadable, MLWritable, MLWriter

logger = logging.getLogger(__name__)

class TmpСommonMLWriter(MLWriter):
    """Implements saving an Estimator/Transformer instance to disk.
    Used when saving a trained pipeline.
    Implements MLWriter.saveImpl(path) method.
    """

    def __init__(self, stage_uid: str):
        super().__init__()
        self._stage_uid = stage_uid

    def saveImpl(self, path: str) -> None:
        logger.info(f"MLWriter.saveImpl() call to save {self._stage_uid} instance")