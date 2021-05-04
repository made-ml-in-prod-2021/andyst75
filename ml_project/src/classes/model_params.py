from dataclasses import dataclass, field, MISSING

from sklearn.base import BaseEstimator

from .transforms_params import TransformParams


@dataclass()
class ModelParams:
    _target_: BaseEstimator = MISSING
    transforms: TransformParams = MISSING
    model_parameters: dict = field(default_factory=dict)
