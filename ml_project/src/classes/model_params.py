from dataclasses import dataclass, field, MISSING

from .transforms_params import TransformParams
from .model_estimator import ModelEstimator


@dataclass()
class ModelParams:
    model: ModelEstimator = MISSING
    transforms: TransformParams = MISSING
    model_parameters: dict = field(default_factory=dict)
