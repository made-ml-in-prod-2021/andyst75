"""
Supply function
"""
import os
import pickle
from typing import NoReturn

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd


def make_path(path: str) -> str:
    """
    Create path from root to target path (fix change path by hydra)
    """
    if os.path.isabs(path) or not HydraConfig.initialized():
        return path

    root = get_original_cwd()
    return os.path.join(root, path)


def dump_object(path: str, obj: object) -> NoReturn:
    """
    Dump object into file
    """
    data_path = make_path(path)
    with open(data_path, "wb") as filename:
        pickle.dump(obj, filename, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(path: str) -> object:
    """
    Load object from file
    """
    data_path = make_path(path)
    with open(data_path, "rb") as filename:
        obj = pickle.load(filename)
    return obj
