""" Get predict by http-request """
import logging
import os
from typing import NoReturn, Optional, Any

import click
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("get_predict")

DEFAULT_SUCCESS_STATUS_CODE = 200


def check_answer(response_data: dict) -> NoReturn:
    """ Check answer for valid """

    if not isinstance(response_data, dict):
        msg_err = f"Wrong response type: {type(response_data)}, expect dict"
        logger.error(msg_err)
        raise RuntimeError(msg_err)

    if "predict" not in response_data:
        msg_err = "Predict data not found in response fata"
        logger.error(msg_err)
        raise RuntimeError(msg_err)

    data = response_data["predict"]
    if not isinstance(data, list):
        msg_err = f"Wrong response data type: {type(data)}, expect list"
        logger.error(msg_err)
        raise RuntimeError(msg_err)

    try:
        predict_data = np.array(data)
    except Exception as error:
        msg_err = "Bad data"
        logger.error(msg_err)
        raise RuntimeError(msg_err) from error

    if len(data) != len(predict_data):
        msg_err = "Bad rows count in predict," +\
                  f"got {predict_data} expect {data}"
        logger.error(msg_err)
        raise RuntimeError(msg_err)


@click.command()
@click.option("-d", "--data-path",
              default=os.path.join("data", "heart.csv"),
              show_default=True)
@click.option("-u", "--request_url",
              default="http://0.0.0.0:8000/predict",
              show_default=True)
def main(data_path: str = None, request_url: str = None) -> Optional[Any]:
    logger.info("Start predict by http-request")

    try:
        data_df = pd.read_csv(data_path)
    except Exception as error:
        msg_err = f"Error read datafile: {str(error)}"
        logger.error(msg_err)
        raise RuntimeError(msg_err) from error

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
        raise RuntimeError(msg_err)

    response_data = response.json()
    check_answer(response_data)

    logger.info("Finish predict by http-request")


if __name__ == "__main__":
    main()
