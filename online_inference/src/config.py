import logging.config
import os
from functools import lru_cache
from typing import Union

from pydantic import BaseSettings
from sklearn.base import BaseEstimator

from src.classes import PredictParams, FeatureParams
from src.data import DatasetTransformer
from src.utils import read_config, load_estimator, load_features

CONFIG_PATH = os.path.join("configs", "app_conf.yaml")

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """ Application setting """
    is_init: bool = False
    config_path: Union[None, str] = None
    app_config: Union[None, PredictParams] = None
    model: Union[None, BaseEstimator] = None
    features: Union[None, FeatureParams] = None
    transformer: Union[None, DatasetTransformer] = None


settings = Settings(config_path=CONFIG_PATH)


@lru_cache()
def get_setting() -> Settings:
    """ Dynamic update config by pytest """
    if settings.config_path is not None:
        logger.info("Load configuration")
        config = read_config(settings.config_path)
        settings.app_config = config

    if not settings.is_init:
        logger.info("Load model")
        settings.model = load_estimator(settings.app_config.model)

        logger.debug("Load features")
        settings.features = load_features(settings.app_config.features)

        settings.transformer = \
            DatasetTransformer(settings.features,
                               transform_path=settings.app_config
                               .transform_path)

        settings.is_init = True

    return settings
