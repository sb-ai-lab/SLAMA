import logging.config


from lightautoml.spark.ml_algo.base import AveragingTransformer
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import logging_config
from lightautoml.spark.utils import spark_session


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    with spark_session(master="local[4]") as spark:

        data = [
            ([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [6.0, 8.0, 9.0]),
            ([float('nan'), 2.0, 3.0], [3.0, float('nan'), 4.0], [6.0, 8.0, 9.0]),
            ([1.0, 2.0, 3.0], [3.0, 2.0, float('nan')], [6.0, float('nan'), 9.0]),
            ([float('nan'), float('nan'), float('nan')], [float('nan'), float('nan'), float('nan')], [float('nan'), float('nan'), float('nan')])
        ]

        pred_cols = ["prediction_1", "prediction_2", "prediction_3"]
        df = spark.createDataFrame(data, pred_cols)
        df.show(truncate=False)

        transformer = AveragingTransformer(
            task_name="multiclass",
            input_cols=pred_cols,
            output_col="blended_prediction",
            remove_cols=pred_cols
        )

        df = transformer.transform(df)
        df.show(truncate=False)

        data = [
            (1.0, 2.0, 3.0),
            (float('nan'), 2.0, 3.0),
            (float('nan'), float('nan'), 3.0),
            (float('nan'), float('nan'), float('nan'))
        ]
        df = spark.createDataFrame(data, pred_cols)
        df.show(truncate=False)
        transformer = AveragingTransformer(
            task_name="reg",
            input_cols=pred_cols,
            output_col="blended_prediction",
            remove_cols=pred_cols
        )
        df = transformer.transform(df)
        df.show(truncate=False)

        logger.info("Finished")
