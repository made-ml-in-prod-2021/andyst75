"""
Module for build and save predict
"""
import logging.config
import os

import click
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from sklearn.base import BaseEstimator

from .classes import PredictParams, FeatureParams, AppResponse
from .data import DatasetTransformer
from .predict import predict, check_request
from .utils import read_config, load_estimator, load_features

APP_CONFIG: PredictParams
MODEL: BaseEstimator
FEATURES: FeatureParams
TRANSFORMER: DatasetTransformer

app = FastAPI()
logger = logging.getLogger("app.main")


@app.post("/predict", response_model=AppResponse)
def app_predict(request: dict):
    """ Predict by request """
    logger.debug('Start request')

    try:
        data, app_request = check_request(request, FEATURES)
    except Exception as error:
        return PlainTextResponse(str(error), status_code=400)

    data_df = pd.DataFrame(data, columns=app_request.features)
    predit_target = predict(MODEL, TRANSFORMER, data_df)

    response = AppResponse(predict=predit_target.tolist())

    return response


@app.get("/")
def app_root():
    """ Root directory """
    print(APP_CONFIG)
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

    global APP_CONFIG, MODEL, FEATURES, TRANSFORMER
    APP_CONFIG = config

    logger.info("Load model")
    MODEL = load_estimator(config.model)

    logger.debug("Load features")
    FEATURES = load_features(config.features)

    TRANSFORMER = DatasetTransformer(FEATURES,
                                     transform_path=config.transform_path)

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
