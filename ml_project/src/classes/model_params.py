"""
Dataclass for model parameters (from YAML-file)
"""

from dataclasses import dataclass, field, MISSING

from .transforms_params import TransformParams
from .model_estimator import ModelEstimator


@dataclass()
class ModelParams:
    """
    Dataclass for model name, model  parameters (from YAML-file) and
    categorical/numerical transformation
    """

    model: ModelEstimator = MISSING
    transforms: TransformParams = MISSING
    model_parameters: dict = field(default_factory=dict)
