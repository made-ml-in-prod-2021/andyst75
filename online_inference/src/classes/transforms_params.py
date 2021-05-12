"""
Dataclass for transform estimator and parameters (from YAML-file)
"""

from dataclasses import dataclass, field
from typing import Union

from .transform_estimator import TransformEstimator


@dataclass()
class TransformParams:
    """
    Dataclass for transform estimator and parameters
    """

    numerical_transform: Union[None, TransformEstimator] = field(default=None)
    categorical_transform: Union[None, TransformEstimator] = \
        field(default=None)
    numerical_parameters: dict = field(default_factory=dict)
    categorical_parameters: dict = field(default_factory=dict)
