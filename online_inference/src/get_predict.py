""" Get predict by http-request """
import logging
import os
import sys
from typing import NoReturn

import click
import numpy as np
import pandas as pd
import requests

from src.classes import HttpPredictResponse

logger = logging.getLogger(__name__)

DEFAULT_SUCCESS_STATUS_CODE = 200
READ_DATA_ERROR = 1
INCORRECT_RESPONSE = 2


def check_request_answer(response_data: dict, data_lenght: int) -> NoReturn:
    if not isinstance(response_data, dict):
        msg_err = f"Wrong response type: {type(response_data)}, expect dict"
        logger.error(msg_err)
        sys.exit(READ_DATA_ERROR)

    try:
        response_struct = HttpPredictResponse(**response_data)
    except ValueError as error:
        msg_err = "Predict data not found in response: " + str(error)
        logger.error(msg_err)
        sys.exit(INCORRECT_RESPONSE)

    data = response_struct.predict

    try:
        _ = np.array(data)
    except Exception as error:
        msg_err = "Bad data:" + str(error)
        logger.error(msg_err)
        sys.exit(INCORRECT_RESPONSE)

    if len(data) != data_lenght:
        msg_err = "Bad rows count in predict," +\
                  f"got {len(data)} expect {data_lenght}"
        logger.error(msg_err)
        sys.exit(INCORRECT_RESPONSE)


@click.command()
@click.option("-d", "--data-path",
              default=os.path.join("data", "heart.csv"),
              show_default=True)
@click.option("-u", "--request_url",
              default="http://0.0.0.0:8000/predict",
              show_default=True)
def main(data_path: str = None, request_url: str = None):
    logger.info("Start predict by http-request")

    try:
        data_df = pd.read_csv(data_path)
    except Exception as error:
        msg_err = f"Error read datafile: {str(error)}"
        logger.error(msg_err)
        sys.exit(READ_DATA_ERROR)

    features = data_df.columns.tolist()
    data = data_df.values.tolist()

    logger.debug("Request predict")

    response = requests.post(
        request_url,
        json={"features": features, "data": data, },
    )

    logger.debug("Response http-code: %s", str(response.status_code))

    if response.status_code != DEFAULT_SUCCESS_STATUS_CODE:
        msg_err = f"Wrong response code: {response.status_code}"
        logger.error(msg_err)
        sys.exit(INCORRECT_RESPONSE)

    response_data = response.json()
    check_request_answer(response_data, len(data))

    logger.info("Finish predict by http-request")


if __name__ == "__main__":
    main()
