"""
Module for build and save predict
"""
import logging.config
import os
import time
import datetime
from typing import Any, Optional

import click
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse

from src.classes import HttpPredictResponse, HttpPredictRequest
from src.predict import predict
from src.utils import read_config
from src.config import CONFIG_PATH, settings, get_setting, Settings

OK_STATUS_CODE = 200
DEFAULT_VALIDATION_ERROR_CODE = 400

TIME_SLEEP = 23
TIME_FAIL = 127

app = FastAPI()
logger = logging.getLogger("app.main")
start_time = datetime.datetime.now()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(response, exc):
    return PlainTextResponse(str(exc),
                             status_code=DEFAULT_VALIDATION_ERROR_CODE)


@app.on_event("startup")
def prepare_model():
    time.sleep(TIME_SLEEP)


@app.post("/predict", response_model=HttpPredictResponse)
def app_predict(request: HttpPredictRequest,
                config: Settings = Depends(get_setting)):

    """ Predict by request """
    logger.debug('Start request')

    data_df = pd.DataFrame(request.data_np, columns=request.features)
    predit_target = predict(settings.model, config.transformer, data_df)

    response = HttpPredictResponse(predict=predit_target.tolist())

    return response


@app.get("/")
def app_root(config: Settings = Depends(get_setting)):
    """ Root directory """
    logger.info(config.config_path)

    return "Prediction Heart Disease UCI"


@app.get("/healthz")
def health() -> JSONResponse:

    now = datetime.datetime.now()
    elapsed_time = now - start_time
    if elapsed_time.seconds > TIME_FAIL:
        raise Exception("App is dead by timeout")

    return JSONResponse(
        status_code=OK_STATUS_CODE,
        content=(not (settings.model is None)),
    )


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
