"""
Utils for working with dataset
"""
import logging
import os
from typing import Tuple, Set

import pandas as pd
from sklearn.model_selection import train_test_split

from ..classes import SplittingParams, FeatureParams
from ..utils import make_path

logger = logging.getLogger("data.dataset")


def read_data(path: str) -> pd.DataFrame:
    """ Read source data in pandas dataframe """
    logger.info("Start reading data from %s", path)
    data_path = make_path(path)

    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        error_msg = f"Data path not exists: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Finish read data from %s", path)
    return data


def check_data(data_df: pd.DataFrame, features: FeatureParams) -> \
        (bool, Set[str], Set[str]):
    """ Check dataframe for contains some features """

    logger.info("Start check data")
    columns_set = set(data_df.columns.tolist())
    categorical = set(features.categorical_features) - columns_set
    numerical = set(features.numerical_features) - columns_set

    check_result = len(categorical) == 0 and len(numerical) == 0

    logger.info("Result check data: %s", str(check_result))

    return (check_result,
            categorical,
            numerical)


def split_train_val_data(data: pd.DataFrame, params: SplittingParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split pandas dataframe to training and validate data """

    logger.info("Start splitting data: %s", str(data.shape))
    train_data, val_data = train_test_split(
        data,
        test_size=params.val_size,
        random_state=params.random_state
    )
    logger.info("Finish splitting data: train=%s, validate=%s",
                str(train_data.shape), str(val_data.shape))
    return train_data, val_data
