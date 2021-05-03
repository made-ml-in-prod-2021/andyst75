from dataclasses import dataclass, MISSING
from sklearn.base import BaseEstimator


@dataclass()
class TransformEstimator:
    _target_: BaseEstimator = MISSING
