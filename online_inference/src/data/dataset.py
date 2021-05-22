"""
Utils for working with dataset
"""
import logging
from typing import Set, List, Tuple

import pandas as pd

from src.classes import FeatureParams


logger = logging.getLogger("data.dataset")


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
