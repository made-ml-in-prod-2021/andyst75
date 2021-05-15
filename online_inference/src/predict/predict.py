import logging.config

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.data import DatasetTransformer

logger = logging.getLogger("app.predict")


def predict(model: BaseEstimator,
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
