"""
Dataclass for split dataframe parameters (from YAML-file)
"""

from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    """
    Dataclass for split dataframe (sklearn train_test_split)
    """

    val_size: float = field(default=0.2)
    random_state: int = field(default=42)
