"""
Utilities for predict
"""

import os
import pickle
import yaml

from sklearn.base import BaseEstimator
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from ..classes import PredictParams, FeatureParams, TransformPath


def make_path(path: str) -> str:
    """
    Create path from root to target path (fix change path by hydra)
    """
    if os.path.isabs(path) or not HydraConfig.initialized():
        return path

    root = get_original_cwd()
    return os.path.join(root, path)


def read_config(config_path: str,
                data_path: str = None,
                output_path: str = None) -> PredictParams:
    """ Read main predict config file """
    with open(config_path, "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    config = PredictParams(**yaml_config)

    if isinstance(config.transform_path, dict):
        config.transform_path = TransformPath(**config.transform_path)

    if data_path:
        config.data_path = data_path
    if output_path:
        config.output_path = output_path

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
