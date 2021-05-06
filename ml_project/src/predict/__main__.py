"""
Module for build and save predict
"""
import logging.config
import os
from typing import NoReturn

import click
import pandas as pd
import yaml

from ..data.data_transformer import DatasetTransformer
from .predict_utils import read_config, load_estimator, load_features

logger = logging.getLogger("predict.main")


@click.command()
@click.option("-c", "--config-path",
              default="configs/predict.yaml",
              required=True, show_default=True)
@click.option("-d", "--data-path", default=None)
@click.option("-o", "--output-path", default=None)
def predict(config_path=None, data_path=None, output_path=None) -> NoReturn:
    """
    Predict-pipeline.
    Required config_path.
    Optional data_path, output_path (overwrite parameters from config)
    """
    config = read_config(config_path, data_path, output_path)

    if os.path.exists(config.log_config):
        with open(config.log_config, "r") as log_config:
            log_config = yaml.safe_load(log_config)
            logging.config.dictConfig(log_config)

    logger.info("Start predict")

    logger.debug("Load model")
    model = load_estimator(config.model)

    logger.debug("Load features")
    features = load_features(config.features)

    transformer = DatasetTransformer(features,
                                     transform_path=config.transform_path)

    logger.debug("Load data")
    data_df = pd.read_csv(config.data_path)
    logger.info("Loaded %d rows", data_df.shape[0])

    logger.debug("Transform data")
    x_data = transformer.transform(data_df)

    logger.debug("Predict")
    y_predict = model.predict(x_data)

    logger.debug("Export predict to %s", config.predict_path)
    pd.Series(y_predict, index=data_df.index, name="prediction") \
        .to_csv(config.predict_path)

    logger.info("Finish predict")


if __name__ == "__main__":
    predict()
