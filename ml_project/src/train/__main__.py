"""
Main module for train models
"""
import logging.config
import os
from typing import Tuple, NoReturn

import hydra
import numpy as np
import pandas as pd
import yaml

from ..utils import make_path
from ..train import build_train_report
from ..classes import ConfigParams
from ..data import read_data, check_data, split_train_val_data
from ..data.data_transformer import DatasetTransformer
from ..model import build_model, dump_model

logger = logging.getLogger("train.main")


def get_data(cfg: ConfigParams) -> \
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Load and transform data for train """
    data_df = read_data(cfg.input_data_path)

    check_df, cat_error, num_error = check_data(data_df, cfg.features)

    if not check_df:
        error_msg = f"Some features not found: { {*cat_error, *num_error} }"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    train_df, test_df = split_train_val_data(data_df, cfg.split)

    ft_transforms = DatasetTransformer(cfg.features,
                                       trans_param=cfg.models.transforms)
    ft_transforms.fit(train_df)
    ft_transforms.dump(cfg.transform_path)

    x_train = ft_transforms.transform(train_df)
    x_test = ft_transforms.transform(test_df)

    y_train = train_df[cfg.features.target_col].values
    y_test = test_df[cfg.features.target_col].values

    return data_df, x_train, y_train, x_test, y_test


def train_pipeline(cfg: ConfigParams) -> NoReturn:
    """
    Main train pipeline.
    Parameters read from YAML-file.
    For run with custom parameters usage --config-name=config_name and
    --config-path=config_path (absolute or relative)
    """

    log_path = make_path(cfg.log_path)
    if os.path.exists(log_path):
        with open(log_path, "r") as log_config:
            log_config = yaml.safe_load(log_config)
            logging.config.dictConfig(log_config)

    logger.info("Start train pipeline")

    data_df, x_train, y_train, x_test, y_test = get_data(cfg)

    model = build_model(cfg.models)
    model.fit(x_train, y_train)
    dump_model(cfg.model_path, model)

    report = build_train_report(model, x_test, y_test, data_df, cfg)

    report_text = \
        f"Accuracy: {report.accuracy:.4f}, F1 metric: {report.f1_metric:.4f}"
    logger.info(report_text)

    logger.info("Finish train pipeline")


@hydra.main(config_path=os.path.join("..", "..", "configs"),
            config_name="train")
def main(cfg: ConfigParams = None):
    """ Proxy for main train pipeline """
    train_pipeline(cfg)


if __name__ == '__main__':
    main()
