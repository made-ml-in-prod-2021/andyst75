# pylint: disable=no-name-in-module
""" Dataclasses for Request/Response """

from typing import List, Union

import numpy as np
from pydantic import BaseModel, validator, root_validator

from src.config import get_setting
from src.data import check_features


class AppRequest(BaseModel):
    """ Dataclass for http-request """
    features: List[str]
    data: List[List[Union[int, float, str]]]
    data_np: np.ndarray = None

    @validator('data')
    def validate_data(cls, values: list) -> list:
        if len(values) == 0:
            raise ValueError("Emptys data")
        return values

    @validator('features')
    def validate_features(cls, values: list) -> list:
        config = get_setting()
        check_result, categorical, numerical = \
            check_features(values,
                           config.features.categorical_features,
                           config.features.numerical_features)
        if not check_result:
            msg_err = f"Not found features: {categorical | numerical}"
            raise ValueError(msg_err)
        return values

    @validator('data_np', pre=True)
    def validate_data_np(cls, values: np.ndarray) -> np.ndarray:
        if values is None:
            return None

        if len(values) == 0:
            raise ValueError("Emptys data")

        return values

    @root_validator()
    def check_data_features(cls, values: dict) -> dict:

        try:
            data = np.array(values.get("data"))
        except Exception as error:
            msg_err = "Incorrect data structure"
            raise ValueError(msg_err) from error

        if (len(data.shape) != 2) or \
                (data.shape[1] != len(values.get("features"))):
            msg_err = "Feature columns and Data columns is different"
            raise ValueError(msg_err)

        values["data_np"] = data

        return values

    class Config:
        arbitrary_types_allowed = True


class AppResponse(BaseModel):
    """ Dataclass for http-response """
    predict: List[int]
