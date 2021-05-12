"""
Module for build and save predict
"""
import logging.config
import os
from typing import List

import click
import numpy as np
import pandas as pd
import yaml
import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator
from pydantic import parse_obj_as

from .classes import PredictParams, FeatureParams
from .data import DatasetTransformer, check_features
from .predict import predict

from .classes import AppResponse, AppRequest
from .utils import read_config, load_estimator, load_features

app = FastAPI()
app_config: PredictParams
model: BaseEstimator
features: FeatureParams
transformer: DatasetTransformer

logger = logging.getLogger("app.main")


@app.post("/predict", response_model=AppResponse)
def app_predict(request: dict):
    logger.debug('Start request')

    try:
        app_request = AppRequest(**request)
    except Exception as e:
        msg_err = f"Bad request structure: {str(e)}"
        logger.error(msg_err)
        return PlainTextResponse(msg_err, status_code=400)

    check_result, categorical, numerical = check_features(
        app_request.features,
        features.categorical_features,
        features.numerical_features)

    if not check_result:
        err_msg = f"Not found features: {categorical | numerical}"
        return PlainTextResponse(err_msg, status_code=400)

    if len(app_request.data) == 0:
        err_msg = "Empty data"
        logger.error(err_msg)
        return PlainTextResponse(err_msg, status_code=400)

    try:
        data = np.array(app_request.data)
    except Exception as e:
        logger.error(str(e))
        return PlainTextResponse(str(e), status_code=400)

    if data.shape[1] != len(app_request.features):
        err_msg = f"Feature columns and Data columns is different"
        logger.error(err_msg)
        return PlainTextResponse(err_msg, status_code=400)

    data_df = pd.DataFrame(data, columns=app_request.features)
    predit_target = predict(app_config, model, features, transformer, data_df)

    response = AppResponse(predict=predit_target.tolist())

    return response


@app.get("/")
def app_root():
    print(app_config)
    return "Prediction Heart Disease UCI"


@click.command()
@click.option("-c", "--config-path",
              default=os.path.join("configs", "app_conf.yaml"),
              required=True, show_default=True)
@click.option("-h", "--host", default=None)
@click.option("-p", "--port", default=None)
def main(config_path=None, host=None, port=None):

    config = read_config(config_path, host, port)

    if os.path.exists(config.log_config):
        with open(config.log_config, "r") as log_config:
            log_config = yaml.safe_load(log_config)
            logging.config.dictConfig(log_config)

    logger.info("Running app on %s with port %s", config.host, config.port)

    global app_config, model, features, transformer

    app_config = config

    logger.info("Load model")
    model = load_estimator(config.model)

    logger.debug("Load features")
    features = load_features(config.features)

    transformer = DatasetTransformer(features,
                                     transform_path=config.transform_path)

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
