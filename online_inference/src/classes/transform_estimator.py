"""
Dataclass for categorical/numerical transform estimator
"""

from dataclasses import dataclass, MISSING
from sklearn.base import BaseEstimator


@dataclass()
class TransformEstimator:
    """
    Dataclass for create instance of transform estimator
    """

    _target_: BaseEstimator
