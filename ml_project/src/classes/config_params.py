"""
Dataclass for config parameters (from YAML-file)
"""

from dataclasses import dataclass, field, MISSING

from .model_params import ModelParams
from .feature_params import FeatureParams
from .split_params import SplittingParams
from .transform_path import TransformPath


@dataclass()
class ConfigParams:
    """
    Dataclass for YAML config
    """

    models: ModelParams = MISSING
    features: FeatureParams = MISSING
    split: SplittingParams = MISSING

    input_data_path: str = field(default="data/raw/heart.csv")
    model_path: str = field(default="models/model.pkl")
    report_path: str = field(default="reports/train.yaml")
    log_path: str = field(default="configs/log_config.yaml")

    transform_path: TransformPath = field(default_factory=TransformPath)
