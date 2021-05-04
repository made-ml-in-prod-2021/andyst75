"""
Dataclass for create model training report
"""

import datetime
from dataclasses import dataclass, field, MISSING

from .split_params import SplittingParams
from .model_params import ModelParams


@dataclass()
class ModelReport:
    """
    Dataclass for create model training report
    """

    model: ModelParams = MISSING
    split: SplittingParams = MISSING
    data: str = field(default="data.csv")
    rows: int = field(default=0)
    accuracy: float = 0.
    f1_metric: float = 0.
    date: str = field(default=datetime.datetime.now().isoformat())
