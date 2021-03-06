"""
Utils for build, load and dump model
"""
import logging
from typing import NoReturn

import hydra
from sklearn.base import BaseEstimator

from ..classes import ModelParams
from ..utils import load_object, dump_object

logger = logging.getLogger("model.model_utils")


def build_model(model_params: ModelParams) -> BaseEstimator:
    """ Create instance of model with parameters """

    model_name = model_params.model['_target_']
    model_parameters = model_params.model_parameters
    logger.info("Build model %s", model_name)
    logger.info("Parameters: %s", model_parameters)

    model = hydra.utils.instantiate(model_params.model,
                                    **model_params.model_parameters)
    return model


def load_model(path: str) -> BaseEstimator:
    """ Load fitted model from pickle file """
    logger.info("Loading model")
    model = load_object(path)
    return model


def dump_model(path: str, model: BaseEstimator) -> NoReturn:
    """ Dump fitted model to pickle file """
    logger.info("Dumping model")
    dump_object(path, model)
