"""
Utilities for predict
"""

import pickle
import yaml

from sklearn.base import BaseEstimator

from src.classes import PredictParams, FeatureParams, TransformPath


def read_config(config_path: str,
                host: str = None,
                port: str = None) -> PredictParams:
    """ Read main predict config file """
    with open(config_path, "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    config = PredictParams(**yaml_config)

    if isinstance(config.transform_path, dict):
        config.transform_path = TransformPath(**config.transform_path)

    if host:
        config.host = host
    if port:
        config.port = port

    return config


def load_estimator(path: str) -> BaseEstimator:
    """ Load estimator (model, transformers) from pickle-file. """
    with open(path, "rb") as pickle_file:
        estimator = pickle.load(pickle_file)
    return estimator


def load_features(path: str) -> FeatureParams:
    """ Load features from YAML-file """
    with open(path, "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    features = FeatureParams(**yaml_config)

    return features
