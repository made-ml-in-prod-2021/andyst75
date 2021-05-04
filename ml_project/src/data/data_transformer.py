"""
Module for data transform
"""
from typing import NoReturn
import pickle
import logging

import pandas as pd
import numpy as np

import hydra

from sklearn.base import BaseEstimator, TransformerMixin

from ..classes import TransformParams
from ..classes import FeatureParams
from ..classes import TransformPath
from ..utils import make_path

logger = logging.getLogger("data.data_transformer")


def _dump_file(path: str, obj: object) -> NoReturn:
    data_path = make_path(path)
    with open(data_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_dump(path: str) -> object:
    data_path = make_path(path)
    with open(data_path, "rb") as f:
        obj = pickle.load(f)
    return obj


class DatasetTransformer(BaseEstimator, TransformerMixin):
    """
    Transform categorical and numerical data.
    E.g. scikit-learn OneHotEncoder for categorical data and
    StandartScaler numerical data
    """

    def __init__(self,
                 feature_param: FeatureParams,
                 trans_param: TransformParams,
                 transform_path: TransformPath = None):
        """
        Load fitted transform if TransformPath is not None
        """

        logger.info("Start init transforms")

        self.categorical_features = feature_param.categorical_features
        self.numerical_features = feature_param.numerical_features

        if transform_path:
            logger.debug("Loading transforms")
            self.categorical = _load_dump(transform_path.categorical)
            self.numerical = _load_dump(transform_path.numerical)
        else:
            logger.debug("Create instance transforms")
            self.categorical = \
                hydra.utils.instantiate(trans_param.categorical_transform,
                                        **trans_param.categorical_parameters)
            self.numerical = \
                hydra.utils.instantiate(trans_param.numerical_transform,
                                        **trans_param.numerical_parameters)

        logger.info("Finish init transforms")

    def fit(self, data: pd.DataFrame) -> NoReturn:
        """
        Fit transforms on data
        """

        logger.info("Start fitting data")

        logger.debug("Start fitting categorical data")
        self.categorical.fit(data[self.categorical_features].values)
        logger.debug("Finish fitting categorical data")

        logger.debug("Start fitting numerical data")
        self.numerical.fit(data[self.numerical_features].values)
        logger.debug("Finish fitting numerical data")

        logger.info("Finish fitting data")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data
        """

        logger.info("Start transform data")

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

    def dump(self, transform_path: TransformPath) \
            -> NoReturn:
        """
        Dump fitting transformers
        """

        logger.info("Start dump transform data")

        logger.debug("Start dump numerical transform data")
        _dump_file(transform_path.numerical, self.numerical)
        logger.debug("Finish dump numerical transform data")

        logger.debug("Start dump categorical transform data")
        _dump_file(transform_path.categorical, self.categorical)
        logger.debug("Finish dump categorical transform data")

        logger.info("Finish dump transform data")
