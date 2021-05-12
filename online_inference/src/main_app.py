"""
Module for build and save predict
"""
import logging.config
import os
from functools import lru_cache
from typing import Any, Optional, Union

import click
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseSettings
from sklearn.base import BaseEstimator

from .classes import PredictParams, FeatureParams, AppResponse
from .data import DatasetTransformer
from .predict import predict, check_request
from .utils import read_config, load_estimator, load_features

DEFAULT_VALIDATION_ERROR_CODE = 400
CONFIG_PATH = os.path.join("configs", "app_conf.yaml")

app = FastAPI()
logger = logging.getLogger("app.main")


class Settings(BaseSettings):
    is_init: bool = False
    config_path: Union[None, str] = None
    app_config: Union[None, PredictParams] = None
    model: Union[None, BaseEstimator] = None
    features: Union[None, FeatureParams] = None
    transformer: Union[None, DatasetTransformer] = None


settings = Settings(config_path=CONFIG_PATH)


@lru_cache()
def get_setting() -> Settings:
    """ Dynamic update config by pytest """
    if settings.config_path is not None:
        logger.info("Load configuration")
        config = read_config(settings.config_path)
        settings.app_config = config

    if not settings.is_init:
        logger.info("Load model")
        settings.model = load_estimator(settings.app_config.model)

        logger.debug("Load features")
        settings.features = load_features(settings.app_config.features)

        settings.transformer = \
            DatasetTransformer(settings.features,
                               transform_path=settings.app_config
                               .transform_path)

        settings.is_init = True

    return settings


@app.post("/predict", response_model=AppResponse)
def app_predict(request: dict, config: Settings = Depends(get_setting)):
    """ Predict by request """
    logger.debug('Start request')

    print("predict")

    try:
        data, app_request = check_request(request, config.features)
    except Exception as error:
        logger.error(str(error))
        return PlainTextResponse(str(error),
                                 status_code=DEFAULT_VALIDATION_ERROR_CODE)

    data_df = pd.DataFrame(data, columns=app_request.features)
    predit_target = predict(settings.model, config.transformer, data_df)

    response = AppResponse(predict=predit_target.tolist())

    return response


@app.get("/")
def app_root(config: Settings = Depends(get_setting)):
    """ Root directory """
    print(config.config_path)

    return "Prediction Heart Disease UCI"


@click.command()
@click.option("-c", "--config-path",
              default=CONFIG_PATH,
              required=True, show_default=True)
@click.option("-h", "--host", default=None)
@click.option("-p", "--port", default=None)
def main(config_path=None, host=None, port=None) -> Optional[Any]:
    config = read_config(config_path, host, port)
    settings.app_config = config

    if os.path.exists(config.log_config):
        with open(config.log_config, "r") as log_config:
            log_config = yaml.safe_load(log_config)
            logging.config.dictConfig(log_config)

    logger.info("Running app on %s with port %s", config.host, config.port)

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
