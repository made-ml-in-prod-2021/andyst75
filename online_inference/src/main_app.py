"""
Module for build and save predict
"""
import logging.config
import os
from typing import Any, Optional

import click
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from src.classes import AppResponse, AppRequest
from src.predict import predict
from src.utils import read_config
from src.config import CONFIG_PATH, settings, get_setting, Settings

DEFAULT_VALIDATION_ERROR_CODE = 400

app = FastAPI()
logger = logging.getLogger("app.main")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(response, exc):
    return PlainTextResponse(str(exc),
                             status_code=DEFAULT_VALIDATION_ERROR_CODE)


@app.post("/predict", response_model=AppResponse)
def app_predict(request: AppRequest, config: Settings = Depends(get_setting)):
    """ Predict by request """
    logger.debug('Start request')

    data_df = pd.DataFrame(request.data_np, columns=request.features)
    predit_target = predict(settings.model, config.transformer, data_df)

    response = AppResponse(predict=predit_target.tolist())

    return response


@app.get("/")
def app_root(config: Settings = Depends(get_setting)):
    """ Root directory """
    logger.info(config.config_path)

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
