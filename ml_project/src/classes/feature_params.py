from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    target_col: str = field(default="target")
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
