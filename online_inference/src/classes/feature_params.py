"""
Dataclass for feature parameters (from YAML-file)
"""

from dataclasses import dataclass, field
from typing import List, Union


@dataclass()
class FeatureParams:
    """
    Dataclass for feature parameters
    """

    target_col: str = field(default="target")
    categorical_features: Union[None, List[str]] = field(default=None)
    numerical_features: Union[None, List[str]] = field(default=None)
