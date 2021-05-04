from dataclasses import dataclass, MISSING
from sklearn.base import BaseEstimator


@dataclass()
class ModelEstimator:
    _target_: BaseEstimator = MISSING
