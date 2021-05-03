from dataclasses import dataclass, field, MISSING

from .model_params import ModelParams
from .feature_params import FeatureParams
from .split_params import SplittingParams


@dataclass()
class ConfigParams:
    models: ModelParams = MISSING
    features: FeatureParams = MISSING
    split: SplittingParams = MISSING

    input_data_path: str = field(default="data/raw/heart.csv")
    output_model_path: str = field(default="models/model.pkl")
    metric_path: str = field(default="models/metrics.json")
