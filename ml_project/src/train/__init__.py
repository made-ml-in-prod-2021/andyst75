""" Init for training package """
from .make_report import build_train_report
from .__main__ import train_pipeline

__all__ = ["build_train_report", "train_pipeline"]
