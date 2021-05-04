import datetime
from dataclasses import dataclass, field, MISSING
from typing import Dict, Any

import numpy as np

from ..classes import SplittingParams, ModelParams


@dataclass()
class ModelReport(Dict[str, Any]):
    model: ModelParams = MISSING
    split: SplittingParams = MISSING
    data: str = field(default="data.csv")
    rows: int = field(default=0)
    accuracy: float = 0.
    f1_metric: float = 0.
    date: str = field(default=datetime.datetime.now().isoformat())
