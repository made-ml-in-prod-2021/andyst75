import numpy as np
import pandas as pd

import logging.config

from sklearn.base import BaseEstimator
from src.classes import FeatureParams

from ..classes import PredictParams

from ..data import DatasetTransformer

logger = logging.getLogger("app.predict")


def predict(config: PredictParams,
            model: BaseEstimator,
            features: FeatureParams,
            transformer: DatasetTransformer,
            data_df: pd.DataFrame) -> np.ndarray:
    """ Predict function. """

    logger.info("Start predict")

    logger.info("Got %d rows", data_df.shape[0])

    logger.debug("Transform data")
    x_data = transformer.transform(data_df)

    logger.debug("Predict")
    y_pred = model.predict(x_data)

    logger.info("Finish predict")

    return y_pred