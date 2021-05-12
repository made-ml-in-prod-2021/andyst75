"""
Utils for working with dataset
"""
import logging
import os
from typing import Set, List, Tuple

import pandas as pd

from src.classes import FeatureParams
from src.utils import make_path

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


def check_features(data_features: List[str],
                   categorical_features: List[str],
                   numerical_features: List[str]) -> Tuple[bool, set, set]:
    """ Check features """

    columns_set = set(data_features)
    categorical = set(categorical_features) - columns_set
    numerical = set(numerical_features) - columns_set

    check_result = len(categorical) == 0 and len(numerical) == 0

    return (check_result,
            categorical,
            numerical)


def check_data(data_df: pd.DataFrame, features: FeatureParams) -> \
        Tuple[bool, Set[str], Set[str]]:
    """ Check dataframe for contains some features """

    logger.info("Start check data")

    check_result, categorical, numerical = check_features(
        data_df.columns.tolist(),
        features.categorical_features,
        features.numerical_features)

    logger.info("Result check data: %s", str(check_result))

    return (check_result,
            categorical,
            numerical)
