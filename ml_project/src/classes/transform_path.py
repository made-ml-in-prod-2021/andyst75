"""
Dataclass for transform path
"""

from dataclasses import dataclass, field


@dataclass()
class TransformPath:
    """
    Dataclass for transform path
    """

    categorical: str = field(default="categorical.pkl")
    numerical: str = field(default="numerical.pkl")
