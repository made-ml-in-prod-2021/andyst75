import logging
from typing import NoReturn

import hydra
from sklearn.base import BaseEstimator

from ..classes import ModelParams
from ..utils import load_object, dump_object

logger = logging.getLogger("model.model_utils")


def build_model(model_params: ModelParams) -> BaseEstimator:
    logger.info("Build model")
    model = hydra.utils.instantiate(model_params.model,
                                    **model_params.model_parameters)
    return model


def load_model(path: str) -> BaseEstimator:
    logger.info("Loading model")
    model = load_object(path)
    return model


def dump_model(path: str, model: BaseEstimator) -> NoReturn:
    logger.info("Dumping model")
    dump_object(path, model)
