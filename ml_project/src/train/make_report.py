"""
Module for create training report
"""

import logging

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

from ..classes import ModelReport, ConfigParams
from ..utils import make_path

logger = logging.getLogger("train.make_report")


def build_train_report(model: BaseEstimator,
                       x_test: np.ndarray,
                       y_test: np.ndarray,
                       data_df: pd.DataFrame,
                       cfg: ConfigParams) -> ModelReport:
    """ Build and write training report in yaml format """

    logger.info("Start build training report")

    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    f1_metric = f1_score(y_test, y_pred)

    logger.debug("Create model report")
    report = ModelReport(cfg.models, cfg.split)
    report.data = cfg.input_data_path
    report.rows = data_df.shape[0]
    report.accuracy = float(accuracy)
    report.f1_metric = float(f1_metric)

    logger.debug("Convert report to yaml")
    yaml_report = OmegaConf.to_yaml(report)

    logger.debug("Write report")
    with open(make_path(cfg.report_path), "w") as yaml_file:
        yaml_file.writelines(yaml_report)

    logger.info("Finish build training report")

    return report
