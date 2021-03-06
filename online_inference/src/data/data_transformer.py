"""
Module for data transform
"""
import logging

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from src.classes import FeatureParams, TransformPath
from src.utils import load_estimator

logger = logging.getLogger("data.data_transformer")


class DatasetTransformer(BaseEstimator, TransformerMixin):
    """
    Transform categorical and numerical data.
    E.g. scikit-learn OneHotEncoder for categorical data and
    StandartScaler numerical data
    """

    def __init__(self,
                 feature_param: FeatureParams,
                 transform_path: TransformPath = None):
        """
        Load fitted transform if TransformPath is not None
        """

        logger.info("Start init transforms")

        self.categorical_features = feature_param.categorical_features
        self.numerical_features = feature_param.numerical_features

        if transform_path:
            logger.debug("Loading transforms")
            self.categorical = load_estimator(transform_path.categorical)
            self.numerical = load_estimator(transform_path.numerical)
        else:
            msg_error = "Both transform_path and trans_param is None"
            logger.error(msg_error)
            raise NotImplementedError(msg_error)

        logger.info("Finish init transforms")

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data
        """

        logger.debug("Start transform categorical data")
        num_scaler = \
            self.numerical.transform(
                data[self.numerical_features].values)
        logger.debug("Finish transform categorical data")

        logger.debug("Start transform numerical data")
        cat_ohe = \
            self.categorical.transform(
                data[self.categorical_features].values)
        logger.debug("Finish transform numerical data")

        if len(num_scaler) != len(cat_ohe):
            msg_error = "Numerical and Categorical data has difference rows"
            logger.error(msg_error)
            raise RuntimeError(msg_error)

        transformed_data = np.hstack([num_scaler, cat_ohe])
        logger.info("Finish transform data")

        return transformed_data
