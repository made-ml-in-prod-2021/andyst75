import logging.config
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.classes import FeatureParams, AppRequest
from src.data import DatasetTransformer, check_features

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


def check_request(request: dict, features: FeatureParams) \
        -> Tuple[np.ndarray, AppRequest]:
    """ Check http-request """
    try:
        app_request = AppRequest(**request)
    except Exception as error:
        msg_err = f"Bad request structure: {str(error)}"
        logger.error(msg_err)
        raise ValueError(msg_err) from error

    check_result, categorical, numerical = check_features(
        app_request.features,
        features.categorical_features,
        features.numerical_features)

    if not check_result:
        msg_err = f"Not found features: {categorical | numerical}"
        logger.error(msg_err)
        raise ValueError(msg_err)

    if len(app_request.data) == 0:
        msg_err = "Empty data"
        logger.error(msg_err)
        raise ValueError(msg_err)

    try:
        data = np.array(app_request.data)
    except Exception as error:
        msg_err = "Incorrect data structure"
        logger.error(msg_err)
        raise ValueError(msg_err) from error

    if data.shape[1] != len(app_request.features):
        msg_err = "Feature columns and Data columns is different"
        logger.error(msg_err)
        raise ValueError(msg_err)

    return data, app_request
