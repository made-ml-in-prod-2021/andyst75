from dataclasses import dataclass, field


@dataclass()
class TransformPath:
    categorical: str = field(default="categorical.pkl")
    numerical: str = field(default="numerical.pkl")
