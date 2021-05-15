"""
Dataclass for keep model estimator (YAML-file)
"""

from dataclasses import dataclass, MISSING
from sklearn.base import BaseEstimator


@dataclass()
class ModelEstimator:
    """
    Dataclass for create instance of model
    """

    _target_: BaseEstimator = MISSING
