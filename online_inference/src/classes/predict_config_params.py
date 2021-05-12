"""
Dataclass for predict config parameters (from YAML-file)
"""

from dataclasses import dataclass, field

from ..classes import TransformPath


@dataclass()
class PredictParams:
    """
    Dataclass for YAML predict config
    """

    host: str = field(default='0.0.0.0')
    port: int = field(default=8000)

    log_config: str = field(default="configs/log_config.yaml")
    model: str = field(default="models/model.pkl")

    transform_path: TransformPath = None
    features: str = field(default="configs/features/5cat_8num.yaml")

    data_path: str = field(default_factory=str)
