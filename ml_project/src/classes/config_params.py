from dataclasses import dataclass, field, MISSING

from .model_params import ModelParams
from .feature_params import FeatureParams
from .split_params import SplittingParams
from .transform_path import TransformPath


@dataclass()
class ConfigParams:
    models: ModelParams = MISSING
    features: FeatureParams = MISSING
    split: SplittingParams = MISSING

    input_data_path: str = field(default="data/raw/heart.csv")
    model_path: str = field(default="models/model.pkl")

    transform_path: TransformPath = field(default_factory=TransformPath)

    # numerical_transform_path: str = field(default="models/numerical.pkl")
    # categorical_transform_path: str = field(default="models/numerical.pkl")
