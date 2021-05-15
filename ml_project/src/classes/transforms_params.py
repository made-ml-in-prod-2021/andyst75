"""
Dataclass for transform estimator and parameters (from YAML-file)
"""

from dataclasses import dataclass, field, MISSING

from .transform_estimator import TransformEstimator


@dataclass()
class TransformParams:
    """
    Dataclass for transform estimator and parameters
    """

    numerical_transform: TransformEstimator = MISSING
    categorical_transform: TransformEstimator = MISSING
    numerical_parameters: dict = field(default_factory=dict)
    categorical_parameters: dict = field(default_factory=dict)
