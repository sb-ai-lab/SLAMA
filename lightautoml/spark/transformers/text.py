from copy import copy
from typing import List, Optional, Dict, Tuple

import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF, Normalizer

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorRole
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.text import TunableTransformer, text_check


class TfidfTextTransformer(SparkTransformer, TunableTransformer):
    """Simple Tfidf vectorizer."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tfidf"
    _default_params = {
        "min_df": 5,
        "max_df": 1.0,
        "max_features": 30_000,
        "dtype": np.float32,
        "normalization": 2.0
    }

    # These properties are not supported
    # cause there is no analogues in Spark ML
    # "ngram_range": (1, 1),
    # "analyzer": "word",

    @property
    def features(self) -> List[str]:
        """Features list."""

        return self._features

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        subs: Optional[float] = None,
        random_state: int = 42,
    ):
        """

        Args:
            default_params: algo hyperparams.
            freeze_defaults: Flag.
            subs: Subsample to calculate freqs. If ``None`` - full data.
            random_state: Random state to take subsample.

        Note:
            The behaviour of `freeze_defaults`:

            - ``True`` :  params may be rewritten depending on dataset.
            - ``False``:  params may be changed only
              manually or with tuning.

        """
        super().__init__(default_params, freeze_defaults)
        self.subs = subs
        self.random_state = random_state
        self.idf_columns_pipelines: Optional[Dict[str, Tuple[PipelineModel, str, int]]] = None
        self.vocab_size: Optional[int] = None

    def init_params_on_input(self, dataset: SparkDataset) -> dict:
        """Get transformer parameters depending on dataset parameters.

        Args:
            dataset: Dataset used for model parmaeters initialization.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        suggested_params = copy(self.default_params)
        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        # TODO: decide later what to do with this part
        # rows_num = len(dataset.data)
        # if rows_num > 50_000:
        #     suggested_params["min_df"] = 25

        return suggested_params

    def fit(self, dataset: SparkDataset):
        """Fit tfidf vectorizer.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        if self._params is None:
            self.params = self.init_params_on_input(dataset)

        sdf = dataset.data
        if self.subs:
            sdf = sdf.sample(self.subs, seed=self.random_state)
        sdf = sdf.fillna("")

        self.idf_columns_pipelines = dict()
        feats = []
        for c in sdf.columns:
            # TODO: set params here from self.params
            tokenizer = Tokenizer(inputCol=c, outputCol=f"{c}_words")
            count_tf = CountVectorizer(
                minDF=self.params["min_df"],
                maxDF=self.params["max_df"],
                vocabSize=self.params["max_features"],
                inputCol=tokenizer.getOutputCol(),
                outputCol=f"{c}_word_features"
            )
            out_col = f"{self._fname_prefix}__{c}"
            idf = IDF(inputCol=count_tf.getOutputCol(), outputCol=f"{c}_idf_features")

            stages = [tokenizer, count_tf, idf]
            if self.params["normalization"] and self.params["normalization"] > 0:
                norm = Normalizer(inputCol=idf.getOutputCol(), outputCol=out_col, p=self.params["normalization"])
                stages.append(norm)

            pipeline = Pipeline(stages=stages)
            tfidf_pipeline_model = pipeline.fit(sdf)

            # features = list(
            #     np.char.array([self._fname_prefix + "_"])
            #     + np.arange(count_tf.getVocabSize()).astype(str)
            #     + np.char.array(["__" + c])
            # )
            # feats.extend(features)

            self.idf_columns_pipelines[c] = (tfidf_pipeline_model, out_col, count_tf.getVocabSize())

        self._features = feats

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform text dataset to sparse tfidf representation.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Sparse dataset with encoded text.

        """
        # checks here
        super().transform(dataset)

        sdf = dataset.data
        sdf = sdf.fillna("")

        # transform

        curr_sdf = sdf
        all_idf_features = []
        all_idf_roles = []
        for c in sdf.columns:
            tfidf_model, idf_col, vocab_size = self.idf_columns_pipelines[c]

            curr_sdf = tfidf_model.transform(curr_sdf)

            role = NumericVectorRole(
                size=vocab_size,
                element_col_name_template=f"{self._fname_prefix}_{{}}__{idf_col}"
            )

            all_idf_features.append(idf_col)
            all_idf_roles.append(role)

        # all_idf_features = [
        #     vector_to_array(F.col(idf_col))[i].alias(feat)
        #     for idf_col, features in idf2features.items()
        #     for i,feat in enumerate(features)
        # ]

        new_sdf = curr_sdf.select(all_idf_features)

        output = dataset.empty()
        output.set_data(new_sdf, self._features, all_idf_roles)

        return output
